"""Database of inspirational quotes from scientific icons."""

import random
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Quote:
    """A quote from a scientific icon."""

    text: str
    author: str
    field: str  # e.g., "Physics", "Mathematics", "Computer Science"
    era: str  # e.g., "20th Century", "19th Century"
    theme: str  # e.g., "First Principles", "Curiosity", "Simplicity"
    context: Optional[str] = None  # Additional context about the quote


# Curated collection of quotes from scientific icons
QUOTES: list[Quote] = [
    # Richard Feynman - Physics
    Quote(
        text="What I cannot create, I do not understand.",
        author="Richard Feynman",
        field="Physics",
        era="20th Century",
        theme="First Principles",
        context="Found on Feynman's blackboard at the time of his death",
    ),
    Quote(
        text="The first principle is that you must not fool yourself - and you are the easiest person to fool.",
        author="Richard Feynman",
        field="Physics",
        era="20th Century",
        theme="Critical Thinking",
    ),
    Quote(
        text="Study hard what interests you the most in the most undisciplined, irreverent and original manner possible.",
        author="Richard Feynman",
        field="Physics",
        era="20th Century",
        theme="Learning",
    ),
    Quote(
        text="I learned very early the difference between knowing the name of something and knowing something.",
        author="Richard Feynman",
        field="Physics",
        era="20th Century",
        theme="Understanding",
    ),
    Quote(
        text="It doesn't matter how beautiful your theory is. If it doesn't agree with experiment, it's wrong.",
        author="Richard Feynman",
        field="Physics",
        era="20th Century",
        theme="Scientific Method",
    ),
    # Albert Einstein - Physics
    Quote(
        text="If you can't explain it simply, you don't understand it well enough.",
        author="Albert Einstein",
        field="Physics",
        era="20th Century",
        theme="Simplicity",
    ),
    Quote(
        text="Imagination is more important than knowledge. Knowledge is limited. Imagination encircles the world.",
        author="Albert Einstein",
        field="Physics",
        era="20th Century",
        theme="Creativity",
    ),
    Quote(
        text="The important thing is not to stop questioning. Curiosity has its own reason for existing.",
        author="Albert Einstein",
        field="Physics",
        era="20th Century",
        theme="Curiosity",
    ),
    Quote(
        text="A person who never made a mistake never tried anything new.",
        author="Albert Einstein",
        field="Physics",
        era="20th Century",
        theme="Learning",
    ),
    Quote(
        text="Logic will get you from A to B. Imagination will take you everywhere.",
        author="Albert Einstein",
        field="Physics",
        era="20th Century",
        theme="Creativity",
    ),
    # Marie Curie - Chemistry/Physics
    Quote(
        text="Nothing in life is to be feared, it is only to be understood. Now is the time to understand more, so that we may fear less.",
        author="Marie Curie",
        field="Chemistry",
        era="20th Century",
        theme="Understanding",
    ),
    Quote(
        text="Be less curious about people and more curious about ideas.",
        author="Marie Curie",
        field="Chemistry",
        era="20th Century",
        theme="Curiosity",
    ),
    Quote(
        text="I was taught that the way of progress was neither swift nor easy.",
        author="Marie Curie",
        field="Chemistry",
        era="20th Century",
        theme="Perseverance",
    ),
    # Alan Turing - Computer Science
    Quote(
        text="We can only see a short distance ahead, but we can see plenty there that needs to be done.",
        author="Alan Turing",
        field="Computer Science",
        era="20th Century",
        theme="Progress",
    ),
    Quote(
        text="Sometimes it is the people no one can imagine anything of who do the things no one can imagine.",
        author="Alan Turing",
        field="Computer Science",
        era="20th Century",
        theme="Potential",
    ),
    # Claude Shannon - Information Theory
    Quote(
        text="Information is the resolution of uncertainty.",
        author="Claude Shannon",
        field="Information Theory",
        era="20th Century",
        theme="First Principles",
    ),
    # Isaac Newton - Physics/Mathematics
    Quote(
        text="If I have seen further, it is by standing on the shoulders of giants.",
        author="Isaac Newton",
        field="Physics",
        era="17th Century",
        theme="Building on Knowledge",
    ),
    Quote(
        text="I do not know what I may appear to the world, but to myself I seem to have been only like a boy playing on the seashore.",
        author="Isaac Newton",
        field="Physics",
        era="17th Century",
        theme="Humility",
    ),
    # Ada Lovelace - Computer Science
    Quote(
        text="The Analytical Engine has no pretensions to originate anything. It can do whatever we know how to order it to perform.",
        author="Ada Lovelace",
        field="Computer Science",
        era="19th Century",
        theme="Computing",
        context="First description of computer programming",
    ),
    # Nikola Tesla - Electrical Engineering
    Quote(
        text="The scientists of today think deeply instead of clearly. One must be sane to think clearly, but one can think deeply and be quite insane.",
        author="Nikola Tesla",
        field="Electrical Engineering",
        era="20th Century",
        theme="Clear Thinking",
    ),
    Quote(
        text="If you want to find the secrets of the universe, think in terms of energy, frequency and vibration.",
        author="Nikola Tesla",
        field="Electrical Engineering",
        era="20th Century",
        theme="First Principles",
    ),
    # Carl Sagan - Astronomy
    Quote(
        text="Somewhere, something incredible is waiting to be known.",
        author="Carl Sagan",
        field="Astronomy",
        era="20th Century",
        theme="Discovery",
    ),
    Quote(
        text="The cosmos is within us. We are made of star-stuff. We are a way for the universe to know itself.",
        author="Carl Sagan",
        field="Astronomy",
        era="20th Century",
        theme="Connection",
    ),
    # Emmy Noether - Mathematics
    Quote(
        text="My methods are really methods of working and thinking; this is why they have crept in everywhere anonymously.",
        author="Emmy Noether",
        field="Mathematics",
        era="20th Century",
        theme="Method",
        context="On her fundamental contributions to abstract algebra",
    ),
    # John von Neumann - Mathematics/Computing
    Quote(
        text="In mathematics you don't understand things. You just get used to them.",
        author="John von Neumann",
        field="Mathematics",
        era="20th Century",
        theme="Learning",
    ),
    # Edsger Dijkstra - Computer Science
    Quote(
        text="Simplicity is prerequisite for reliability.",
        author="Edsger Dijkstra",
        field="Computer Science",
        era="20th Century",
        theme="Simplicity",
    ),
    Quote(
        text="The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise.",
        author="Edsger Dijkstra",
        field="Computer Science",
        era="20th Century",
        theme="Abstraction",
    ),
    # Donald Knuth - Computer Science
    Quote(
        text="Premature optimization is the root of all evil.",
        author="Donald Knuth",
        field="Computer Science",
        era="20th Century",
        theme="Engineering",
    ),
    # Grace Hopper - Computer Science
    Quote(
        text="The most dangerous phrase in the language is: We've always done it this way.",
        author="Grace Hopper",
        field="Computer Science",
        era="20th Century",
        theme="Innovation",
    ),
    Quote(
        text="A ship in port is safe, but that's not what ships are built for.",
        author="Grace Hopper",
        field="Computer Science",
        era="20th Century",
        theme="Risk",
    ),
    # Werner Heisenberg - Physics
    Quote(
        text="What we observe is not nature itself, but nature exposed to our method of questioning.",
        author="Werner Heisenberg",
        field="Physics",
        era="20th Century",
        theme="Observation",
    ),
    # Niels Bohr - Physics
    Quote(
        text="An expert is a person who has made all the mistakes that can be made in a very narrow field.",
        author="Niels Bohr",
        field="Physics",
        era="20th Century",
        theme="Expertise",
    ),
    Quote(
        text="The opposite of a correct statement is a false statement. But the opposite of a profound truth may well be another profound truth.",
        author="Niels Bohr",
        field="Physics",
        era="20th Century",
        theme="Paradox",
    ),
    # Stephen Hawking - Physics
    Quote(
        text="Intelligence is the ability to adapt to change.",
        author="Stephen Hawking",
        field="Physics",
        era="21st Century",
        theme="Adaptability",
    ),
    Quote(
        text="However difficult life may seem, there is always something you can do and succeed at.",
        author="Stephen Hawking",
        field="Physics",
        era="21st Century",
        theme="Perseverance",
    ),
    # Rosalind Franklin - Chemistry/Biology
    Quote(
        text="Science and everyday life cannot and should not be separated.",
        author="Rosalind Franklin",
        field="Chemistry",
        era="20th Century",
        theme="Science in Life",
    ),
    # Linus Torvalds - Computer Science
    Quote(
        text="Talk is cheap. Show me the code.",
        author="Linus Torvalds",
        field="Computer Science",
        era="21st Century",
        theme="Action",
    ),
]


class QuoteDatabase:
    """Manager for the quotes collection."""

    def __init__(self, quotes: Optional[list[Quote]] = None):
        """Initialize with quotes collection.

        Args:
            quotes: Optional custom quotes list. Uses default QUOTES if not provided.
        """
        self.quotes = quotes or QUOTES

    def get_random(self) -> Quote:
        """Get a random quote."""
        return random.choice(self.quotes)

    def get_by_author(self, author: str) -> list[Quote]:
        """Get all quotes by a specific author."""
        return [q for q in self.quotes if author.lower() in q.author.lower()]

    def get_by_theme(self, theme: str) -> list[Quote]:
        """Get all quotes with a specific theme."""
        return [q for q in self.quotes if theme.lower() in q.theme.lower()]

    def get_by_field(self, field: str) -> list[Quote]:
        """Get all quotes from a specific field."""
        return [q for q in self.quotes if field.lower() in q.field.lower()]

    def get_daily(self, seed_date: Optional[date] = None) -> Quote:
        """Get a deterministic quote for a given date.

        Uses the date as a seed so the same quote is returned for the same date.

        Args:
            seed_date: Date to use as seed. Defaults to today.

        Returns:
            Quote: The quote for that date.
        """
        if seed_date is None:
            seed_date = date.today()

        # Use date as seed for reproducibility
        seed = seed_date.year * 10000 + seed_date.month * 100 + seed_date.day
        random.seed(seed)
        quote = random.choice(self.quotes)
        random.seed()  # Reset to random state
        return quote

    def get_inaugural(self) -> Quote:
        """Get a special inaugural quote (Feynman's famous blackboard quote)."""
        for quote in self.quotes:
            if "cannot create" in quote.text.lower() and quote.author == "Richard Feynman":
                return quote
        # Fallback
        return self.get_by_author("Feynman")[0]

    @property
    def authors(self) -> list[str]:
        """Get list of unique authors."""
        return sorted(set(q.author for q in self.quotes))

    @property
    def themes(self) -> list[str]:
        """Get list of unique themes."""
        return sorted(set(q.theme for q in self.quotes))

    @property
    def fields(self) -> list[str]:
        """Get list of unique fields."""
        return sorted(set(q.field for q in self.quotes))

    def __len__(self) -> int:
        """Return number of quotes."""
        return len(self.quotes)
