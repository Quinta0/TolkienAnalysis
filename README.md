# Tolkien Literary Analysis Tool

A comprehensive Python application for analyzing J.R.R. Tolkien's literary works through character network analysis, chapter structure visualization, vocabulary progression tracking, and keyword frequency analysis.

## Features

### Character Co-occurrence Network Analysis
- Identifies character, place, and artifact mentions throughout the text
- Creates interactive network visualizations showing relationships between entities
- Uses page-based co-occurrence analysis for more accurate relationship mapping
- Bubble sizes represent total mentions; line thickness indicates relationship strength
- Color-coded entity types (characters, places, artifacts, races)

### Chapter Structure Analysis
- Automatic chapter detection using multiple pattern recognition approaches
- Horizontal stacked bar visualization showing relative chapter lengths
- Gray gradient coloring where darker segments indicate longer chapters
- Interactive hover information displaying chapter titles and word counts

### Vocabulary Progression Tracking
- Analyzes the introduction of new vocabulary throughout the text
- Highlights sections with high linguistic novelty (25%+ new words) in red
- Tracks cumulative vocabulary growth
- Useful for identifying complex or world-building heavy sections

### Keyword Frequency Analysis
- Search functionality for specific terms across chapters
- Dual metrics: absolute count and frequency per 1000 words
- Visual representation of keyword usage patterns
- Suggested keywords for common Tolkien themes

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Dependencies
```bash
pip install streamlit networkx plotly pandas numpy
```

### Optional Dependencies
For PDF support:
```bash
pip install PyPDF2
```

## Usage

### Running the Application
```bash
streamlit run tolkien_analyzer.py
```

The application will open in your default web browser at `http://localhost:8501`

### Input Methods
The tool supports multiple input formats:

1. **Text Files (Recommended)**: Upload `.txt` files for best accuracy
2. **PDF Files**: Upload `.pdf` files (may have entity recognition issues)
3. **Direct Text Input**: Paste text directly into the interface
4. **Sample Text**: Use built-in Hobbit excerpt for demonstration

### Analysis Workflow
1. Choose your input method from the sidebar
2. Configure analysis settings (minimum mentions, co-occurrence thresholds)
3. Click "Run Complete Analysis"
4. Explore results across five tabs:
   - Network Analysis
   - Chapter Analysis
   - Vocabulary Progression
   - Keyword Search
   - Summary

## Supported Works

The tool is optimized for analyzing Tolkien's major works:
- The Silmarillion
- The Hobbit
- The Lord of the Rings trilogy
- Other Middle-earth writings

The entity recognition system includes over 100 characters, places, artifacts, and groups from Tolkien's legendarium.

## Technical Details

### Entity Recognition
- Uses word boundary matching to prevent false positives
- Comprehensive entity dictionary covering major Tolkien works
- Text cleaning to remove publication artifacts and illustrations credits

### Network Analysis
- Built on NetworkX for graph computation
- Force-directed layout algorithms for optimal visualization
- Filtering options to focus on significant relationships

### Chapter Detection
- Multiple regex patterns for different chapter formats
- Fallback to artificial divisions if no chapters detected
- Word count and character count metrics

### Visualization
- Interactive Plotly charts for all visualizations
- Responsive design with hover information
- Professional color schemes and layouts

## File Structure

```
tolkien_analyzer.py    # Main application file
requirements.txt       # Python dependencies
README.md             # This documentation
```

## Known Limitations

1. **PDF Processing**: PDF text extraction can be inconsistent and may cause entity recognition errors. Text files are strongly recommended.

2. **Entity Recognition**: The tool uses a predefined entity list. Obscure characters or alternate spellings may not be detected.

3. **Chapter Detection**: Works best with standard chapter formatting. Complex or non-standard structures may require manual preprocessing.

4. **Performance**: Large texts (100,000+ words) may take several minutes to process completely.

## Tips for Best Results

- **Use plain text files** whenever possible for maximum accuracy
- **Ensure consistent chapter formatting** (e.g., "Chapter 1", "Book I") for automatic detection
- **Adjust filtering parameters** in the sidebar to focus on the most significant relationships
- **Clean your text** of headers, footers, and publication information before analysis

