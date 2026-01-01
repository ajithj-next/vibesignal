# VibeSignal

> First Principles reasoning. Good Vibes. Sublime Visual thinking.

Convert Jupyter notebooks containing first principles reasoning into engaging Twitter/X threads using AI.

## Features

- **Automated Analysis**: Uses Claude AI to extract first principles reasoning and key insights from notebooks
- **Visual Integration**: Automatically extracts and includes matplotlib/plotly visualizations
- **Smart Thread Structure**: Generates engaging, tweet-sized content with optimal flow
- **Multiple Output Formats**: JSON, text, and markdown outputs
- **Beautiful CLI**: Rich terminal interface with progress indicators
- **Configurable**: Flexible configuration via environment variables or CLI options

## Installation

```bash
# Clone the repository
git clone https://github.com/ajithj-next/vibesignal.git
cd vibesignal

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

1. **Set up your API key**:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key
# VIBESIGNAL_ANTHROPIC_API_KEY=your_key_here
```

2. **Convert a notebook**:

```bash
vibesignal convert your_notebook.ipynb
```

That's it! Your thread will be generated as `your_notebook_thread.json`.

## Usage

### Basic Conversion

```bash
# Convert notebook to JSON (default)
vibesignal convert notebook.ipynb

# Specify output file and format
vibesignal convert notebook.ipynb --output thread.txt --format text

# Generate markdown output
vibesignal convert notebook.ipynb --format markdown
```

### Advanced Options

```bash
# Limit number of tweets
vibesignal convert notebook.ipynb --max-tweets 15

# Specify image directory
vibesignal convert notebook.ipynb --image-dir ./my_images

# Use specific API key
vibesignal convert notebook.ipynb --api-key sk-ant-...
```

### Configuration

Show current configuration:

```bash
vibesignal config-show
```

## Configuration Options

VibeSignal can be configured via environment variables (in `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBESIGNAL_ANTHROPIC_API_KEY` | Anthropic API key (required) | - |
| `VIBESIGNAL_CLAUDE_MODEL` | Claude model to use | `claude-sonnet-4-5-20250929` |
| `VIBESIGNAL_MAX_TWEETS` | Maximum tweets in thread | `10` |
| `VIBESIGNAL_MAX_TWEET_LENGTH` | Maximum characters per tweet | `275` |
| `VIBESIGNAL_INCLUDE_THREAD_NUMBERING` | Add (1/N) numbering | `true` |
| `VIBESIGNAL_OUTPUT_FORMAT` | Default output format | `json` |
| `VIBESIGNAL_IMAGE_OUTPUT_DIR` | Image output directory | `images` |

## Output Formats

### JSON

Structured data with full metadata:

```json
{
  "tweets": [
    {
      "position": 1,
      "text": "Ever wondered how k-NN works? Let's explore from first principles (1/8)",
      "image_filename": null,
      "character_count": 72
    },
    {
      "position": 2,
      "text": "First principle: Distance is meaning. Similar items cluster in feature space.",
      "image_filename": "knn_plot.png",
      "character_count": 78
    }
  ],
  "metadata": {
    "source_notebook": "knn_analysis.ipynb",
    "total_tweets": 8,
    "total_images": 3,
    "claude_model": "claude-sonnet-4-5-20250929"
  },
  "hook": "Ever wondered how k-NN works?"
}
```

### Text

Human-readable format with clear structure:

```
============================================================
TWITTER THREAD
============================================================

Source: knn_analysis.ipynb
Generated: 2026-01-01T10:30:00Z
Total tweets: 8
Total images: 3

------------------------------------------------------------

HOOK: Ever wondered how k-NN works?

------------------------------------------------------------

[Tweet 1/8]
Let's explore k-NN from first principles
(45 characters)

[Tweet 2/8]
[IMAGE: knn_plot.png]
First principle: Distance is meaning...
(78 characters)
```

### Markdown

Ready for documentation or blog posts:

```markdown
# Twitter Thread

**Source:** knn_analysis.ipynb
**Total Tweets:** 8

## Hook

> Ever wondered how k-NN works?

## Thread

### Tweet 1/8

Let's explore k-NN from first principles

### Tweet 2/8

![Image](knn_plot.png)

First principle: Distance is meaning...
```

## How It Works

1. **Parse Notebook**: Extracts markdown cells (reasoning) and code cells (with visualizations)
2. **Extract Images**: Saves matplotlib, plotly, and other visualizations from outputs
3. **Analyze with Claude**: AI identifies first principles reasoning and key insights
4. **Generate Thread**: Structures insights into tweet-sized chunks with optimal flow
5. **Format Output**: Creates the final thread in your chosen format

## Example Workflow

```bash
# 1. Write your analysis in a Jupyter notebook
# Include markdown cells explaining first principles
# Add visualizations using matplotlib, plotly, etc.

# 2. Convert to thread
vibesignal convert my_analysis.ipynb

# 3. Review the output
cat my_analysis_thread.json

# 4. Post to Twitter
# Use the JSON output with Twitter API or manually post tweets
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vibesignal

# Run specific test file
pytest tests/test_parsing/test_notebook.py -v
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## Architecture

```
vibesignal/
├── models/          # Data models (Pydantic)
├── parsing/         # Notebook parsing & image extraction
├── analysis/        # Claude AI integration
├── generation/      # Thread generation & formatting
├── output/          # Output writers (JSON, text, markdown)
└── cli.py           # Command-line interface
```

## Requirements

- Python 3.12+
- Anthropic API key (Claude)
- Jupyter notebook files (.ipynb)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Roadmap

- [ ] PDF support for academic papers
- [ ] Custom infographic generation
- [ ] AI image generation integration
- [ ] Web interface
- [ ] Direct Twitter API posting
- [ ] Support for more visualization libraries
- [ ] Batch processing

## Credits

Created by Ajith J

Powered by:
- [Claude](https://anthropic.com/claude) by Anthropic
- [Click](https://click.palletsprojects.com/) for CLI
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [Pydantic](https://pydantic.dev/) for data validation

---

**First Principles reasoning. Good Vibes. Sublime Visual thinking.**
