"""Twitter thread publishing functionality."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tweepy

from vibesignal.models import Thread


class TwitterPublisher:
    """Publish threads to Twitter/X.

    Handles authentication, thread posting with proper reply chaining,
    image uploads, and thread deletion.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_token_secret: str,
        bearer_token: Optional[str] = None,
    ):
        """Initialize Twitter publisher.

        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            access_token: Twitter access token
            access_token_secret: Twitter access token secret
            bearer_token: Optional bearer token for API v2
        """
        # Initialize API v1.1 (for media upload)
        auth = tweepy.OAuth1UserHandler(
            api_key, api_secret, access_token, access_token_secret
        )
        self.api_v1 = tweepy.API(auth)

        # Initialize API v2 (for tweet posting)
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )

        self._username: Optional[str] = None

    def publish_thread(
        self,
        thread: Thread,
        workspace: Path,
        dry_run: bool = False,
        delay_seconds: float = 2.0,
        save_publish_record: bool = True,
    ) -> list[dict]:
        """Publish a thread to Twitter.

        Args:
            thread: Thread to publish
            workspace: Workspace containing images
            dry_run: If True, don't actually post (for testing)
            delay_seconds: Delay between tweets to avoid rate limits
            save_publish_record: Save publish record to workspace for later deletion

        Returns:
            list[dict]: List of published tweet data with IDs and URLs

        Raises:
            tweepy.TweepyException: If posting fails
        """
        published_tweets = []
        previous_tweet_id = None

        # Get username for URLs
        if not dry_run and not self._username:
            user_info = self.verify_credentials()
            self._username = user_info["username"]

        username = self._username or "user"

        for tweet in thread.tweets:
            # Prepare media if tweet has an image
            media_id = None
            if tweet.image_filename:
                image_path = workspace / "images" / tweet.image_filename
                if image_path.exists():
                    media_id = self._upload_image(image_path, dry_run=dry_run)

            # Post tweet
            if dry_run:
                tweet_data = {
                    "id": f"dry_run_{tweet.position}",
                    "text": tweet.text,
                    "media_id": media_id,
                    "url": f"https://twitter.com/{username}/status/dry_run_{tweet.position}",
                    "position": tweet.position,
                }
            else:
                response = self._post_tweet(
                    text=tweet.text,
                    reply_to=previous_tweet_id,
                    media_ids=[media_id] if media_id else None,
                )
                tweet_data = {
                    "id": response.data["id"],
                    "text": tweet.text,
                    "url": f"https://twitter.com/{username}/status/{response.data['id']}",
                    "position": tweet.position,
                }
                previous_tweet_id = response.data["id"]

                # Rate limiting delay
                if tweet.position < len(thread.tweets):
                    time.sleep(delay_seconds)

            published_tweets.append(tweet_data)

        # Save publish record for later deletion
        if save_publish_record and not dry_run:
            self._save_publish_record(workspace, published_tweets, username)

        return published_tweets

    def _save_publish_record(
        self, workspace: Path, published_tweets: list[dict], username: str
    ) -> Path:
        """Save publish record to workspace for later deletion.

        Args:
            workspace: Workspace path
            published_tweets: List of published tweet data
            username: Twitter username

        Returns:
            Path: Path to saved record
        """
        record = {
            "published_at": datetime.now(timezone.utc).isoformat(),
            "username": username,
            "tweet_count": len(published_tweets),
            "tweets": published_tweets,
            "first_tweet_url": published_tweets[0]["url"] if published_tweets else None,
        }

        record_path = workspace / "publish_record.json"
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

        return record_path

    def delete_tweet(self, tweet_id: str) -> bool:
        """Delete a single tweet.

        Args:
            tweet_id: ID of the tweet to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            tweepy.TweepyException: If deletion fails
        """
        try:
            self.client.delete_tweet(tweet_id)
            return True
        except tweepy.NotFound:
            # Tweet already deleted or doesn't exist
            return True
        except tweepy.Forbidden as e:
            raise tweepy.TweepyException(f"Cannot delete tweet (not owned?): {e}") from e

    def delete_thread(
        self,
        workspace: Path,
        delay_seconds: float = 1.0,
    ) -> dict:
        """Delete a published thread using the saved publish record.

        Args:
            workspace: Workspace containing publish_record.json
            delay_seconds: Delay between deletions to avoid rate limits

        Returns:
            dict: Deletion results with success/failure counts

        Raises:
            FileNotFoundError: If no publish record exists
        """
        record_path = workspace / "publish_record.json"
        if not record_path.exists():
            raise FileNotFoundError(
                f"No publish record found at {record_path}. "
                "Thread may not have been published or record was deleted."
            )

        with open(record_path) as f:
            record = json.load(f)

        results = {
            "total": len(record["tweets"]),
            "deleted": 0,
            "failed": 0,
            "already_deleted": 0,
            "errors": [],
        }

        # Delete in reverse order (newest first)
        for tweet_data in reversed(record["tweets"]):
            tweet_id = tweet_data["id"]

            # Skip dry run tweets
            if str(tweet_id).startswith("dry_run_"):
                results["already_deleted"] += 1
                continue

            try:
                if self.delete_tweet(tweet_id):
                    results["deleted"] += 1
                else:
                    results["already_deleted"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"id": tweet_id, "error": str(e)})

            # Rate limiting delay
            time.sleep(delay_seconds)

        # Archive the publish record
        if results["deleted"] > 0 or results["already_deleted"] == results["total"]:
            archive_path = workspace / "publish_record_deleted.json"
            record["deleted_at"] = datetime.now(timezone.utc).isoformat()
            record["deletion_results"] = results
            with open(archive_path, "w") as f:
                json.dump(record, f, indent=2)
            record_path.unlink()

        return results

    def get_publish_record(self, workspace: Path) -> Optional[dict]:
        """Get the publish record for a workspace if it exists.

        Args:
            workspace: Workspace path

        Returns:
            dict: Publish record or None if not found
        """
        record_path = workspace / "publish_record.json"
        if not record_path.exists():
            return None

        with open(record_path) as f:
            return json.load(f)

    def _upload_image(self, image_path: Path, dry_run: bool = False) -> Optional[str]:
        """Upload an image to Twitter.

        Args:
            image_path: Path to image file
            dry_run: If True, don't actually upload

        Returns:
            str: Media ID if successful, None otherwise
        """
        if dry_run:
            return f"dry_run_media_{image_path.stem}"

        try:
            media = self.api_v1.media_upload(filename=str(image_path))
            return media.media_id_string
        except Exception as e:
            print(f"Warning: Failed to upload image {image_path}: {e}")
            return None

    def _post_tweet(
        self,
        text: str,
        reply_to: Optional[str] = None,
        media_ids: Optional[list[str]] = None,
    ) -> tweepy.Response:
        """Post a single tweet.

        Args:
            text: Tweet text
            reply_to: Tweet ID to reply to (for threading)
            media_ids: List of media IDs to attach

        Returns:
            tweepy.Response: Twitter API response
        """
        kwargs = {"text": text}

        if reply_to:
            kwargs["in_reply_to_tweet_id"] = reply_to

        if media_ids:
            kwargs["media_ids"] = media_ids

        return self.client.create_tweet(**kwargs)

    def verify_credentials(self) -> dict:
        """Verify Twitter credentials are valid.

        Returns:
            dict: User information if successful

        Raises:
            tweepy.TweepyException: If credentials are invalid
        """
        user = self.client.get_me()
        return {
            "id": user.data.id,
            "username": user.data.username,
            "name": user.data.name,
        }
