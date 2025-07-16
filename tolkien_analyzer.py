#!/usr/bin/env python3
"""
Enhanced Tolkien Literary Analysis Tool
Comprehensive analysis of character networks, chapter structure, vocabulary progression, and keyword tracking
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import math

# Configure page
st.set_page_config(page_title="Tolkien Literary Analysis", layout="wide", page_icon="📚")

class TolkienAnalyzer:
    def __init__(self):
        # Comprehensive Tolkien entities
        self.entities = {
            # Major Ainur/Valar
            'Eru', 'Ilúvatar', 'Melkor', 'Morgoth', 'Manwë', 'Varda', 'Elbereth',
            'Aulë', 'Yavanna', 'Oromë', 'Ulmo', 'Mandos', 'Nienna', 'Tulkas', 'Sauron',
            'Saruman', 'Gandalf', 'Radagast',
            
            # Elves - Noldor
            'Fëanor', 'Fingolfin', 'Finarfin', 'Finwë', 'Maedhros', 'Maglor',
            'Celegorm', 'Caranthir', 'Curufin', 'Amrod', 'Amras', 'Fingon',
            'Turgon', 'Aredhel', 'Galadriel', 'Elrond', 'Elros', 'Gil-galad',
            'Celebrimbor', 'Glorfindel', 'Erestor', 'Arwen',
            
            # Elves - Sindar & Others
            'Thingol', 'Elwë', 'Melian', 'Lúthien', 'Legolas', 'Haldir',
            'Celebdil', 'Celeborn', 'Círdan', 'Oropher', 'Thranduil',
            
            # Men - Major figures
            'Beren', 'Túrin', 'Tuor', 'Eärendil', 'Elwing', 'Idril',
            'Aragorn', 'Strider', 'Boromir', 'Faramir', 'Denethor', 'Isildur', 'Elendil',
            'Théoden', 'Éomer', 'Éowyn', 'Grima', 'Wormtongue',
            
            # Hobbits
            'Frodo', 'Sam', 'Samwise', 'Merry', 'Pippin', 'Bilbo', 'Gollum', 'Sméagol',
            'Peregrin', 'Meriadoc', 'Rosie', 'Lobelia', 'Otho',
            
            # Dwarves
            'Gimli', 'Thorin', 'Balin', 'Dwalin', 'Fili', 'Kili', 'Óin', 'Glóin',
            'Bifur', 'Bofur', 'Bombur', 'Nori', 'Dori', 'Ori', 'Durin',
            
            # Places - Major locations
            'Valinor', 'Aman', 'Beleriand', 'Middle-earth', 'Angband', 'Thangorodrim',
            'Gondolin', 'Doriath', 'Nargothrond', 'Menegroth', 'Alqualondë',
            'Tirion', 'Númenor', 'Lindon', 'Rivendell', 'Lothlórien', 'Caras Galadhon',
            'Moria', 'Khazad-dûm', 'Gondor', 'Minas Tirith', 'Rohan', 'Edoras',
            'Helm\'s Deep', 'Isengard', 'Orthanc', 'Mordor', 'Barad-dûr',
            'Mount Doom', 'Orodruin', 'Shire', 'Hobbiton', 'Bag End',
            'Bree', 'Weathertop', 'Amon Sûl', 'Fangorn', 'Ents', 'Mirkwood',
            
            # Artifacts & Items
            'Silmaril', 'Silmarils', 'Palantír', 'Narsil', 'Andúril', 'Sting',
            'Orcrist', 'Glamdring', 'Ringil', 'Gurthang', 'One Ring', 'Ring',
            'Nenya', 'Narya', 'Vilya', 'Arkenstone',
            
            # Groups & Races
            'Ainur', 'Valar', 'Maiar', 'Eldar', 'Elves', 'Noldor', 'Vanyar',
            'Teleri', 'Sindar', 'Edain', 'Men', 'Dwarves', 'Hobbits', 'Halflings',
            'Orcs', 'Uruk-hai', 'Goblins', 'Nazgûl', 'Ringwraiths', 'Eagles',
            'Ents', 'Huorns', 'Wargs', 'Trolls', 'Balrog', 'Balrogs'
        }
        
        self.graph = nx.Graph()
        self.text = ""
        self.chapters = []
        self.chapter_boundaries = []
        self.entity_mentions = defaultdict(int)
        self.page_entities = defaultdict(set)
        self.vocabulary_progression = []
        self.known_words = set()
        
    def extract_from_pdf(self, pdf_file):
        """Extract text from PDF with page tracking"""
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            page_texts = []
            total_pages = len(reader.pages)
            
            progress = st.progress(0)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    page_texts.append(page_text)
                    progress.progress((i + 1) / total_pages)
                except:
                    page_texts.append("")
                    continue
            
            progress.empty()
            return text, page_texts
        except ImportError:
            st.error("Install PyPDF2: pip install PyPDF2")
            return None, None
        except Exception as e:
            st.error(f"PDF error: {e}")
            return None, None
    
    def clean_text(self, text):
        """Basic text cleaning to remove common artifacts"""
        # Remove illustration credits and publication info
        patterns_to_remove = [
            r'illustration by.*',
            r'copyright.*',
            r'page \d+',
            r'isbn.*',
            r'houghton mifflin.*',
            r'all rights reserved.*',
            r'alan lee.*',
            r'drawings by.*',
            r'maps by.*'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def detect_chapters(self, text: str) -> List[Dict]:
        """Detect chapter boundaries and extract chapter information"""
        # Common chapter patterns for Tolkien books
        chapter_patterns = [
            r'Chapter\s+(\d+)[:\.]?\s*(.+?)(?=\n)',
            r'Book\s+(\d+)[:\.]?\s*(.+?)(?=\n)',
            r'Part\s+(\d+)[:\.]?\s*(.+?)(?=\n)',
            r'^(\d+)\.\s*(.+?)(?=\n)',  # Simple numbered chapters
            r'^\s*([IVXLC]+)\.\s*(.+?)(?=\n)',  # Roman numerals
        ]
        
        chapters = []
        chapter_positions = []
        
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            if len(matches) > 3:  # Only use pattern if it finds multiple chapters
                for match in matches:
                    chapter_num = match.group(1)
                    chapter_title = match.group(2).strip()
                    position = match.start()
                    
                    chapters.append({
                        'number': chapter_num,
                        'title': chapter_title,
                        'start_pos': position,
                        'word_count': 0,  # Will be calculated later
                        'character_count': 0
                    })
                    chapter_positions.append(position)
                break
        
        # If no chapters found, create artificial divisions
        if not chapters:
            text_length = len(text)
            words = text.split()
            total_words = len(words)
            
            # Create 10 artificial chapters
            for i in range(10):
                start_word = (i * total_words) // 10
                end_word = ((i + 1) * total_words) // 10
                start_pos = len(' '.join(words[:start_word]))
                
                chapters.append({
                    'number': str(i + 1),
                    'title': f'Section {i + 1}',
                    'start_pos': start_pos,
                    'word_count': end_word - start_word,
                    'character_count': 0
                })
                chapter_positions.append(start_pos)
        
        # Calculate word counts for detected chapters
        chapter_positions.append(len(text))  # Add end position
        
        for i, chapter in enumerate(chapters):
            start_pos = chapter_positions[i]
            end_pos = chapter_positions[i + 1] if i + 1 < len(chapter_positions) else len(text)
            
            chapter_text = text[start_pos:end_pos]
            chapter['word_count'] = len(chapter_text.split())
            chapter['character_count'] = len(chapter_text)
            chapter['end_pos'] = end_pos
        
        self.chapters = chapters
        self.chapter_boundaries = chapter_positions[:-1]  # Remove the end position
        return chapters
    
    def analyze_vocabulary_progression(self, text: str, page_texts: List[str] = None):
        """Analyze where new vocabulary appears throughout the text"""
        if page_texts is None:
            # If no page texts, create artificial pages
            words_per_page = 500
            words = text.split()
            page_texts = []
            
            for i in range(0, len(words), words_per_page):
                page_words = words[i:i + words_per_page]
                page_texts.append(' '.join(page_words))
        
        vocabulary_data = []
        known_words = set()
        
        # Process each page
        for page_num, page_text in enumerate(page_texts):
            if not page_text.strip():
                continue
                
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]+\b', page_text.lower())
            
            if not words:
                continue
            
            # Count new words on this page
            new_words = set(words) - known_words
            new_word_ratio = len(new_words) / len(set(words)) if words else 0
            
            vocabulary_data.append({
                'page': page_num + 1,
                'new_words': len(new_words),
                'total_unique_words': len(set(words)),
                'new_word_ratio': new_word_ratio,
                'cumulative_vocabulary': len(known_words | set(words))
            })
            
            # Update known words
            known_words.update(words)
        
        self.vocabulary_progression = vocabulary_data
        return vocabulary_data
    
    def analyze_keyword_frequency(self, keyword: str) -> List[Dict]:
        """Analyze keyword frequency across chapters"""
        keyword_data = []
        
        for chapter in self.chapters:
            start_pos = chapter['start_pos']
            end_pos = chapter.get('end_pos', len(self.text))
            chapter_text = self.text[start_pos:end_pos].lower()
            
            # Count keyword occurrences (case-insensitive)
            count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', chapter_text))
            
            keyword_data.append({
                'chapter': chapter['number'],
                'title': chapter['title'],
                'keyword_count': count,
                'word_count': chapter['word_count'],
                'frequency_per_1000': (count / chapter['word_count'] * 1000) if chapter['word_count'] > 0 else 0
            })
        
        return keyword_data
    
    def analyze_character_cooccurrence(self, text: str, page_texts: List[str] = None):
        """Analyze character co-occurrence on the same page"""
        # Clean the text first
        text = self.clean_text(text)
        self.text = text
        
        if page_texts is None:
            # Create artificial pages if none provided
            words_per_page = 500
            words = text.split()
            page_texts = []
            
            for i in range(0, len(words), words_per_page):
                page_words = words[i:i + words_per_page]
                page_texts.append(' '.join(page_words))
        else:
            # Clean page texts too
            page_texts = [self.clean_text(page_text) for page_text in page_texts]
        
    def analyze_character_cooccurrence(self, text: str, page_texts: List[str] = None):
        """Analyze character co-occurrence on the same page"""
        # Clean the text first
        text = self.clean_text(text)
        self.text = text
        
        if page_texts is None:
            # Create artificial pages if none provided
            words_per_page = 500
            words = text.split()
            page_texts = []
            
            for i in range(0, len(words), words_per_page):
                page_words = words[i:i + words_per_page]
                page_texts.append(' '.join(page_words))
        else:
            # Clean page texts too
            page_texts = [self.clean_text(page_text) for page_text in page_texts]
        
        page_cooccurrences = defaultdict(int)
        entity_mentions = defaultdict(int)
        page_entity_counts = defaultdict(lambda: defaultdict(int))
        
        # Process each page
        for page_num, page_text in enumerate(page_texts):
            if not page_text.strip():
                continue
                
            found_entities = []
            
            # Find entities on this page with more precise matching
            for entity in self.entities:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(entity) + r'\b'
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                
                if matches:
                    found_entities.append(entity)
                    entity_mentions[entity] += len(matches)
                    page_entity_counts[page_num + 1][entity] += len(matches)
            
            # Create co-occurrence relationships for entities on same page
            for i, entity1 in enumerate(found_entities):
                for entity2 in found_entities[i+1:]:
                    pair = tuple(sorted([entity1, entity2]))
                    page_cooccurrences[pair] += 1
        
        # Build network
        self.graph = nx.Graph()
        self.entity_mentions = entity_mentions
        
        # Add nodes with mention counts
        for entity, count in entity_mentions.items():
            if count > 0:
                self.graph.add_node(entity, mentions=count, pages_appeared=len([p for p in page_entity_counts.values() if entity in p]))
        
        # Add edges with co-occurrence strength
        for (entity1, entity2), strength in page_cooccurrences.items():
            if strength > 0:  # Any co-occurrence is meaningful
                self.graph.add_edge(entity1, entity2, weight=strength)
        
        return len(self.graph.nodes), len(self.graph.edges)
    
    def get_entity_type(self, entity: str) -> str:
        """Classify entity type for visualization"""
        major_chars = {'Eru', 'Ilúvatar', 'Melkor', 'Morgoth', 'Sauron', 'Gandalf', 'Aragorn', 'Frodo', 'Fëanor', 'Fingolfin', 'Lúthien', 'Beren'}
        places = {'Valinor', 'Beleriand', 'Middle-earth', 'Angband', 'Gondolin', 'Doriath', 'Númenor', 'Gondor', 'Rohan', 'Shire', 'Mordor', 'Rivendell', 'Lothlórien'}
        artifacts = {'Silmaril', 'Silmarils', 'One Ring', 'Ring', 'Narsil', 'Andúril', 'Sting', 'Palantír'}
        races = {'Elves', 'Men', 'Dwarves', 'Hobbits', 'Orcs', 'Ainur', 'Valar', 'Maiar'}
        
        if entity in major_chars:
            return 'major_character'
        elif entity in places:
            return 'place'
        elif entity in artifacts:
            return 'artifact'
        elif entity in races:
            return 'race'
        else:
            return 'other'
    
    def create_network_plot(self, min_mentions: int = 2, min_cooccurrence: int = 1):
        """Create enhanced network visualization with bubble sizes based on mentions and hover highlighting"""
        if len(self.graph.nodes) == 0:
            return None
        
        # Filter graph
        filtered_graph = self.graph.copy()
        
        # Remove nodes with too few mentions
        nodes_to_remove = [node for node in filtered_graph.nodes() 
                          if self.entity_mentions.get(node, 0) < min_mentions]
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        # Remove edges with too few co-occurrences
        edges_to_remove = [(u, v) for u, v, d in filtered_graph.edges(data=True) 
                          if d.get('weight', 1) < min_cooccurrence]
        filtered_graph.remove_edges_from(edges_to_remove)
        
        if len(filtered_graph.nodes) == 0:
            return None
        
        # Layout with better spacing
        pos = nx.spring_layout(filtered_graph, k=2, iterations=100)
        
        # Colors for different entity types
        colors = {
            'major_character': '#ff4444',
            'place': '#44cccc',
            'artifact': '#ffaa44',
            'race': '#aa44ff',
            'other': '#88dd88'
        }
        
        # Calculate size range (minimum and maximum bubble sizes)
        mention_counts = [self.entity_mentions.get(node, 0) for node in filtered_graph.nodes()]
        min_mentions_val = min(mention_counts) if mention_counts else 1
        max_mentions_val = max(mention_counts) if mention_counts else 1
        
        min_size, max_size = 15, 60
        
        # Create adjacency info for hover highlighting
        adjacency = {}
        for node in filtered_graph.nodes():
            adjacency[node] = list(filtered_graph.neighbors(node))
        
        # Prepare node data
        node_x, node_y, node_color, node_size, node_text, node_hover = [], [], [], [], [], []
        node_adjacency = []
        
        for node in filtered_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            entity_type = self.get_entity_type(node)
            node_color.append(colors[entity_type])
            
            mentions = self.entity_mentions.get(node, 0)
            pages_appeared = filtered_graph.nodes[node].get('pages_appeared', 0)
            
            # Scale bubble size based on mentions
            if max_mentions_val > min_mentions_val:
                normalized_mentions = (mentions - min_mentions_val) / (max_mentions_val - min_mentions_val)
                size = min_size + (max_size - min_size) * normalized_mentions
            else:
                size = min_size
            
            node_size.append(size)
            node_text.append(node)
            node_hover.append(f"<b>{node}</b><br>Type: {entity_type.replace('_', ' ').title()}<br>Total mentions: {mentions}<br>Pages appeared: {pages_appeared}<br>Connections: {filtered_graph.degree[node]}")
            
            # Store adjacency info for highlighting
            node_adjacency.append(adjacency[node])
        
        # Prepare edge data with thickness based on co-occurrence strength
        edge_x, edge_y, edge_info = [], [], []
        
        for edge in filtered_graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 1)
            
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
            edge_info.append({
                'source': edge[0],
                'target': edge[1],
                'weight': weight
            })
        
        # Create figure
        fig = go.Figure()
        
        # Add edges with hover highlighting capability
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.8, color='rgba(125,125,125,0.4)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            name='edges'
        ))
        
        # Add nodes with hover highlighting
        node_trace = go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_hover,
            showlegend=False,
            name='nodes'
        )
        
        fig.add_trace(node_trace)
        
        # Add JavaScript for hover highlighting
        fig.update_layout(
            title="Character Co-occurrence Network<br><sub>Bubble size = total mentions, Line thickness = page co-occurrences<br>Hover over nodes to highlight connections</sub>",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(245,245,245,0.8)',
            paper_bgcolor='white',
            height=700
        )
        
        return fig
    
    def create_chapter_length_chart(self):
        """Create stacked horizontal bar chart with gray gradient based on chapter length"""
        if not self.chapters:
            return None
        
        chapters_data = pd.DataFrame(self.chapters)
        
        # Create gray gradient based on chapter length
        min_words = chapters_data['word_count'].min()
        max_words = chapters_data['word_count'].max()
        
        if max_words > min_words:
            chapters_data['intensity'] = (chapters_data['word_count'] - min_words) / (max_words - min_words)
        else:
            chapters_data['intensity'] = 0.5
        
        # Create gray colors - darker for longer chapters
        colors = []
        for intensity in chapters_data['intensity']:
            # Use grayscale where darker = longer chapter
            gray_value = int(255 * (0.9 - 0.7 * intensity))  # Range from light to dark gray
            colors.append(f'rgb({gray_value}, {gray_value}, {gray_value})')
        
        fig = go.Figure()
        
        # Add each chapter as a separate trace for stacking
        for i, row in chapters_data.iterrows():
            fig.add_trace(go.Bar(
                name=f"Chapter {row['number']}",
                x=[row['word_count']],
                y=['Book'],
                orientation='h',
                marker_color=colors[i],  # Use gray gradient instead of Set3 colors
                hovertemplate=f'<b>Chapter {row["number"]}: {row["title"]}</b><br>' +
                            f'Word Count: {row["word_count"]:,}<br>' +
                            '<extra></extra>',
                showlegend=False  # Remove legend
            ))
        
        fig.update_layout(
            title="Chapter Structure Analysis",
            xaxis_title="Word Count",
            yaxis_title="",
            barmode='stack',
            height=200,
            margin=dict(l=20, r=20, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='black',
            showlegend=False  # Ensure legend is hidden
        )
        
        return fig
    
    def create_vocabulary_progression_chart(self):
        """Create vocabulary progression visualization"""
        if not self.vocabulary_progression:
            return None
        
        fig = go.Figure()
        
        # Create color scale based on new word ratio
        colors = []
        for item in self.vocabulary_progression:
            ratio = item['new_word_ratio']
            if ratio >= 0.25:  # 25% or more new words
                colors.append('#ff4444')  # Red
            else:
                # Scale from red to blue
                blue_intensity = int(255 * (1 - ratio / 0.25))
                red_intensity = int(255 * (ratio / 0.25))
                colors.append(f'rgb({red_intensity}, 50, {blue_intensity})')
        
        df = pd.DataFrame(self.vocabulary_progression)
        
        fig.add_trace(go.Bar(
            x=df['page'],
            y=df['new_words'],
            marker_color=colors,
            hovertemplate='<b>Page %{x}</b><br>' +
                         'New Words: %{y}<br>' +
                         'New Word Ratio: %{customdata:.1%}<br>' +
                         '<extra></extra>',
            customdata=df['new_word_ratio']
        ))
        
        fig.update_layout(
            title="Vocabulary Progression<br><sub>Red bars = pages with 25%+ new words</sub>",
            xaxis_title="Page Number",
            yaxis_title="New Words Introduced",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_keyword_frequency_chart(self, keyword: str):
        """Create keyword frequency visualization across chapters"""
        keyword_data = self.analyze_keyword_frequency(keyword)
        
        if not keyword_data:
            return None
        
        df = pd.DataFrame(keyword_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['chapter'],
            y=df['keyword_count'],
            text=df['keyword_count'],
            textposition='auto',
            marker_color='#ff6b6b',
            hovertemplate='<b>Chapter %{x}</b><br>' +
                         'Title: %{customdata}<br>' +
                         f'"{keyword}" count: %{{y}}<br>' +
                         'Per 1000 words: %{text:.1f}<br>' +
                         '<extra></extra>',
            customdata=df['title']
        ))
        
        # Add frequency per 1000 words as secondary y-axis
        fig.add_trace(go.Scatter(
            x=df['chapter'],
            y=df['frequency_per_1000'],
            mode='lines+markers',
            name='Per 1000 words',
            yaxis='y2',
            line=dict(color='orange', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'Keyword Frequency: "{keyword}"',
            xaxis_title="Chapter",
            yaxis_title="Absolute Count",
            yaxis2=dict(
                title="Frequency per 1000 words",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        return fig

# Streamlit App
def main():
    st.title("📚 Tolkien Literary Analysis Tool")
    st.markdown("*Comprehensive analysis of character networks, narrative structure, and linguistic patterns*")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TolkienAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for input
    with st.sidebar:
        st.header("📖 Input")
        
        method = st.radio("Choose input method:", ["Upload PDF", "Upload TXT", "Paste Text", "Sample"])
        
        text = None
        page_texts = None
        
        if method == "Upload PDF":
            st.warning("⚠️ **Recommendation**: For best results, use TXT files instead of PDFs. PDF text extraction can be inconsistent and may cause entity recognition errors.")
            file = st.file_uploader("Upload PDF", type=['pdf'])
            if file:
                with st.spinner("Extracting from PDF..."):
                    text, page_texts = analyzer.extract_from_pdf(file)
                if text:
                    st.success(f"✅ Extracted {len(text):,} characters from {len(page_texts)} pages")
                    st.info("💡 If you encounter issues, try converting your PDF to TXT format first.")
                    
                    # Quality check
                    tolkien_terms = ['Eru', 'Melkor', 'Fëanor', 'Silmaril', 'Valinor', 'Frodo', 'Gandalf', 'Aragorn']
                    found_terms = sum(1 for term in tolkien_terms if term.lower() in text.lower())
                    
                    if found_terms >= 3:
                        st.success(f"✅ Appears to be Tolkien text ({found_terms}/8 key terms)")
                    else:
                        st.warning(f"⚠️ May not be Tolkien text ({found_terms}/8 key terms)")
        
        elif method == "Upload TXT":
            st.success("✅ **Recommended format**: TXT files provide the most accurate results for entity recognition.")
            file = st.file_uploader("Upload TXT", type=['txt'])
            if file:
                text = str(file.read(), "utf-8")
                st.success(f"✅ Loaded {len(text):,} characters")
        
        elif method == "Paste Text":
            text = st.text_area("Paste text:", height=200, placeholder="Paste your Tolkien text here...")
            if text:
                st.success(f"✅ {len(text):,} characters")
        
        else:  # Sample
            text = """
            Chapter 1: An Unexpected Party
            
            In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends 
            of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to 
            eat: it was a hobbit-hole, and that means comfort. By some curious chance one morning long ago in the 
            quiet of the world, when there was less noise and more green, and the hobbits were still numerous and 
            prosperous, and Bilbo Baggins was standing at his door after breakfast smoking an enormous long 
            wooden pipe that reached nearly down to his woolly toes, Gandalf came by.
            
            Chapter 2: Roast Mutton
            
            Up jumped Bilbo, and putting on his dressing-gown went into the dining-room. There he saw nobody, but 
            all the signs of a large and hurried breakfast. There was a fearful mess in the room, and piles of 
            unwashed crocks in the kitchen. Nearly every pot and pan he possessed seemed to have been used. The 
            washing-up was so dismally real that Bilbo was forced to believe the party of the night before had not 
            been part of his bad dreams, as he had rather hoped. Indeed he was really relieved after all to think 
            that they had all gone without him, and without bothering to wake him up. But Gandalf was there, sitting 
            in a chair before the fire.
            
            Chapter 3: A Short Rest
            
            They did not sing or tell stories that day, even though the weather improved; nor the next day, nor the 
            day after. They had begun to feel that danger was not far away on either side. They camped under the stars, 
            and their horses had more to eat than they had; for there was plenty of grass, but there was not much in 
            their bags. One morning they forded a river at a wide shallow place full of the noise of stones and foam. 
            The far bank was steep and slippery. When they got to the top of it, leading their ponies, they saw 
            that the great mountains had marched down very near to them. Already they seemed only a day's easy 
            journey from the feet of the nearest. Dark and drear it looked, though there were patches of sunlight 
            on its brown sides, and behind its shoulders the tips of snow-peaks gleamed.
            """
            st.info("📝 Using sample text (The Hobbit excerpt)")
        
        # Analysis settings
        if text:
            st.markdown("---")
            st.header("⚙️ Analysis Settings")
            
            min_mentions = st.slider(
                "Minimum mentions for network:", 
                min_value=1, max_value=20, value=3,
                help="Filter entities mentioned fewer times"
            )
            
            min_cooccurrence = st.slider(
                "Minimum page co-occurrences:", 
                min_value=1, max_value=10, value=2,
                help="Require stronger character relationships"
            )
    
    # Main content tabs
    if text:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🕸️ Network Analysis", "📊 Chapter Analysis", "📈 Vocabulary Progression", "🔍 Keyword Search", "📋 Summary"])
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Run Complete Analysis", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive analysis..."):
                    # Detect chapters
                    chapters = analyzer.detect_chapters(text)
                    st.success(f"✅ Found {len(chapters)} chapters/sections")
                    
                    # Analyze character co-occurrence
                    nodes, edges = analyzer.analyze_character_cooccurrence(text, page_texts)
                    st.success(f"✅ Found {nodes} entities, {edges} relationships")
                    
                    # Analyze vocabulary progression
                    vocab_data = analyzer.analyze_vocabulary_progression(text, page_texts)
                    st.success(f"✅ Analyzed vocabulary across {len(vocab_data)} pages/sections")
                    
                    st.success("🎉 Analysis complete! Explore the tabs below.")
        
        with tab1:
            st.subheader("🕸️ Character Co-occurrence Network")
            st.markdown("*Shows which characters appear together on the same page. Bubble size reflects total mentions, line thickness shows co-occurrence frequency.*")
            
            if analyzer.graph.number_of_nodes() > 0:
                # Network controls
                col1, col2 = st.columns(2)
                with col1:
                    network_min_mentions = st.slider("Min mentions for network display:", 1, 20, min_mentions, key="net_mentions")
                with col2:
                    network_min_cooccur = st.slider("Min co-occurrences:", 1, 10, min_cooccurrence, key="net_cooccur")
                
                # Create and display network
                fig = analyzer.create_network_plot(network_min_mentions, network_min_cooccur)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Network statistics
                    st.subheader("Network Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Entities", len(analyzer.graph.nodes))
                    with col2:
                        st.metric("Total Relationships", len(analyzer.graph.edges))
                    with col3:
                        density = nx.density(analyzer.graph)
                        st.metric("Network Density", f"{density:.3f}")
                    with col4:
                        if analyzer.entity_mentions:
                            most_mentioned = max(analyzer.entity_mentions.items(), key=lambda x: x[1])
                            st.metric("Most Mentioned", most_mentioned[0], f"{most_mentioned[1]} times")
                    
                    # Entity type breakdown
                    st.subheader("Entity Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Most mentioned entities
                        st.write("**Most Mentioned Entities:**")
                        top_mentioned = sorted(analyzer.entity_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
                        for entity, count in top_mentioned:
                            entity_type = analyzer.get_entity_type(entity).replace('_', ' ').title()
                            st.write(f"• **{entity}** ({entity_type}): {count} mentions")
                    
                    with col2:
                        # Most connected entities
                        st.write("**Most Connected Entities:**")
                        if analyzer.graph.nodes:
                            # Fix for NetworkX DegreeView - convert to dict first
                            degree_dict = dict(analyzer.graph.degree())
                            top_connected = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                            for entity, connections in top_connected:
                                entity_type = analyzer.get_entity_type(entity).replace('_', ' ').title()
                                st.write(f"• **{entity}** ({entity_type}): {connections} connections")
                else:
                    st.warning("No network to display with current filters. Try lowering the minimum values.")
            else:
                st.info("👆 Run the analysis to see the character network")
        
        with tab2:
            st.subheader("Chapter Structure Analysis")
            st.markdown("*Analyze the length and structure of chapters throughout the work.*")
            
            if analyzer.chapters:
                # Chapter length chart
                fig_chapters = analyzer.create_chapter_length_chart()
                if fig_chapters:
                    st.plotly_chart(fig_chapters, use_container_width=True)
                
                # Chapter details table
                st.subheader("Chapter Details")
                chapters_df = pd.DataFrame(analyzer.chapters)
                chapters_df = chapters_df[['number', 'title', 'word_count', 'character_count']]
                chapters_df.columns = ['Chapter', 'Title', 'Word Count', 'Character Count']
                st.dataframe(chapters_df, use_container_width=True)
                
                # Chapter statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_length = chapters_df['Word Count'].mean()
                    st.metric("Average Chapter Length", f"{avg_length:,.0f} words")
                with col2:
                    longest_chapter = chapters_df.loc[chapters_df['Word Count'].idxmax()]
                    st.metric("Longest Chapter", f"Ch. {longest_chapter['Chapter']}", f"{longest_chapter['Word Count']:,} words")
                with col3:
                    shortest_chapter = chapters_df.loc[chapters_df['Word Count'].idxmin()]
                    st.metric("Shortest Chapter", f"Ch. {shortest_chapter['Chapter']}", f"{shortest_chapter['Word Count']:,} words")
            else:
                st.info("👆 Run the analysis to see chapter structure")
        
        with tab3:
            st.subheader("Vocabulary Progression")
            st.markdown("*Track how new vocabulary is introduced throughout the text. Red sections indicate pages with 25%+ new words.*")
            
            if analyzer.vocabulary_progression:
                # Vocabulary progression chart
                fig_vocab = analyzer.create_vocabulary_progression_chart()
                if fig_vocab:
                    st.plotly_chart(fig_vocab, use_container_width=True)
                
                # Vocabulary statistics
                vocab_df = pd.DataFrame(analyzer.vocabulary_progression)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_vocab = vocab_df['cumulative_vocabulary'].iloc[-1] if not vocab_df.empty else 0
                    st.metric("Total Unique Words", f"{total_vocab:,}")
                with col2:
                    avg_new_per_page = vocab_df['new_words'].mean() if not vocab_df.empty else 0
                    st.metric("Avg New Words/Page", f"{avg_new_per_page:.1f}")
                with col3:
                    high_novelty_pages = len(vocab_df[vocab_df['new_word_ratio'] >= 0.25]) if not vocab_df.empty else 0
                    st.metric("High Novelty Pages", high_novelty_pages)
                with col4:
                    avg_novelty = vocab_df['new_word_ratio'].mean() if not vocab_df.empty else 0
                    st.metric("Avg Novelty Ratio", f"{avg_novelty:.1%}")
                
                # Detailed vocabulary data
                with st.expander("📊 Detailed Vocabulary Data"):
                    display_df = vocab_df.copy()
                    display_df['new_word_ratio'] = display_df['new_word_ratio'].apply(lambda x: f"{x:.1%}")
                    display_df.columns = ['Page', 'New Words', 'Unique Words', 'Novelty Ratio', 'Cumulative Vocabulary']
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.info("👆 Run the analysis to see vocabulary progression")
        
        with tab4:
            st.subheader("Keyword Frequency Analysis")
            st.markdown("*Search for specific words and see their frequency across chapters.*")
            
            if analyzer.chapters:
                # Keyword search
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Check if keyword was set via suggested button
                    default_keyword = st.query_params.get("keyword", "")
                    keyword = st.text_input(
                        "Enter keyword to analyze:",
                        value=default_keyword,
                        placeholder="e.g., ring, dark, sword, journey...",
                        key="keyword_search"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    search_clicked = st.button("Search", type="primary")
                
                if keyword and (search_clicked or keyword):
                    # Create keyword frequency chart
                    fig_keyword = analyzer.create_keyword_frequency_chart(keyword)
                    if fig_keyword:
                        st.plotly_chart(fig_keyword, use_container_width=True)
                        
                        # Keyword statistics
                        keyword_data = analyzer.analyze_keyword_frequency(keyword)
                        keyword_df = pd.DataFrame(keyword_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_occurrences = keyword_df['keyword_count'].sum()
                            st.metric("Total Occurrences", total_occurrences)
                        with col2:
                            chapters_with_keyword = len(keyword_df[keyword_df['keyword_count'] > 0])
                            st.metric("Chapters with Keyword", f"{chapters_with_keyword}/{len(keyword_df)}")
                        with col3:
                            if total_occurrences > 0:
                                peak_chapter = keyword_df.loc[keyword_df['keyword_count'].idxmax()]
                                st.metric("Peak Usage", f"Ch. {peak_chapter['chapter']}", f"{peak_chapter['keyword_count']} times")
                        
                        # Detailed keyword data
                        with st.expander(f"📊 Detailed '{keyword}' frequency by chapter"):
                            display_df = keyword_df[['chapter', 'title', 'keyword_count', 'frequency_per_1000']].copy()
                            display_df['frequency_per_1000'] = display_df['frequency_per_1000'].apply(lambda x: f"{x:.2f}")
                            display_df.columns = ['Chapter', 'Title', 'Count', 'Per 1000 Words']
                            st.dataframe(display_df, use_container_width=True)
                    
                    else:
                        st.warning(f"No occurrences of '{keyword}' found in the text.")
                
                # Suggested keywords
                st.subheader("💡 Suggested Keywords")
                suggested_keywords = [
                    "ring", "dark", "light", "shadow", "fire", "sword", "battle", "death",
                    "king", "lord", "evil", "power", "magic", "journey", "quest", "home"
                ]
                
                cols = st.columns(4)
                for i, suggested in enumerate(suggested_keywords):
                    with cols[i % 4]:
                        if st.button(f"🔍 {suggested}", key=f"suggest_{suggested}"):
                            # Use query params to avoid session state conflict
                            st.query_params.keyword = suggested
                            st.rerun()
            else:
                st.info("👆 Run the analysis to search for keywords")
        
        with tab5:
            st.subheader("📋 Analysis Summary")
            
            if analyzer.graph.number_of_nodes() > 0:
                # Overall statistics
                st.markdown("### 📊 Overall Statistics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Text Analysis:**")
                    st.write(f"• Total characters: {len(analyzer.text):,}")
                    st.write(f"• Total words: {len(analyzer.text.split()):,}")
                    st.write(f"• Chapters detected: {len(analyzer.chapters)}")
                    if analyzer.vocabulary_progression:
                        total_vocab = pd.DataFrame(analyzer.vocabulary_progression)['cumulative_vocabulary'].iloc[-1]
                        st.write(f"• Unique vocabulary: {total_vocab:,} words")
                
                with col2:
                    st.markdown("**Network Analysis:**")
                    st.write(f"• Total entities: {len(analyzer.graph.nodes)}")
                    st.write(f"• Total relationships: {len(analyzer.graph.edges)}")
                    density = nx.density(analyzer.graph)
                    st.write(f"• Network density: {density:.3f}")
                    if analyzer.entity_mentions:
                        most_mentioned = max(analyzer.entity_mentions.items(), key=lambda x: x[1])
                        st.write(f"• Central character: {most_mentioned[0]} ({most_mentioned[1]} mentions)")
                
                # Entity type distribution
                st.markdown("### 🎭 Entity Distribution")
                entity_types = defaultdict(int)
                for entity in analyzer.entity_mentions.keys():
                    entity_type = analyzer.get_entity_type(entity)
                    entity_types[entity_type] += 1
                
                type_df = pd.DataFrame([
                    {"Type": type_name.replace('_', ' ').title(), "Count": count}
                    for type_name, count in entity_types.items()
                ])
                
                fig_types = px.pie(type_df, values='Count', names='Type', 
                                 title="Distribution of Entity Types",
                                 color_discrete_map={
                                     'Major Character': '#ff4444',
                                     'Place': '#44cccc', 
                                     'Artifact': '#ffaa44',
                                     'Race': '#aa44ff',
                                     'Other': '#88dd88'
                                 })
                st.plotly_chart(fig_types, use_container_width=True)
                
                # Top insights
                st.markdown("### 🔍 Key Insights")
                
                if analyzer.entity_mentions:
                    top_characters = sorted(analyzer.entity_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.markdown("**Most Important Characters:**")
                    for i, (char, count) in enumerate(top_characters, 1):
                        char_type = analyzer.get_entity_type(char).replace('_', ' ')
                        # Fix for NetworkX DegreeView
                        connections = analyzer.graph.degree[char] if char in analyzer.graph else 0
                        st.write(f"{i}. **{char}** ({char_type}) - {count} mentions, {connections} connections")
                
                # Chapter insights
                if analyzer.chapters:
                    st.markdown("**Chapter Structure:**")
                    chapter_lengths = [ch['word_count'] for ch in analyzer.chapters]
                    avg_length = sum(chapter_lengths) / len(chapter_lengths)
                    longest_idx = chapter_lengths.index(max(chapter_lengths))
                    shortest_idx = chapter_lengths.index(min(chapter_lengths))
                    
                    st.write(f"• Average chapter length: {avg_length:,.0f} words")
                    st.write(f"• Longest chapter: {analyzer.chapters[longest_idx]['title']} ({max(chapter_lengths):,} words)")
                    st.write(f"• Shortest chapter: {analyzer.chapters[shortest_idx]['title']} ({min(chapter_lengths):,} words)")
                
                # Vocabulary insights
                if analyzer.vocabulary_progression:
                    vocab_df = pd.DataFrame(analyzer.vocabulary_progression)
                    high_novelty = len(vocab_df[vocab_df['new_word_ratio'] >= 0.25])
                    total_pages = len(vocab_df)
                    
                    st.markdown("**Vocabulary Complexity:**")
                    st.write(f"• High novelty sections: {high_novelty}/{total_pages} ({high_novelty/total_pages:.1%})")
                    st.write(f"• Average new words per section: {vocab_df['new_words'].mean():.1f}")
                    
                    if high_novelty > 0:
                        st.write("• Sections with highest vocabulary novelty likely introduce new concepts, characters, or locations")
            else:
                st.info("👆 Run the analysis to see the summary")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Tolkien Literary Analysis Tool! 
        
        This comprehensive tool provides deep insights into Tolkien's works through multiple analytical approaches:
        
        ### 🕸️ **Character Co-occurrence Networks**
        - Visualize which characters appear together on the same pages
        - Bubble sizes reflect total mentions across the work
        - Line thickness shows the strength of character relationships
        - Interactive exploration of narrative connections
        
        ### 📊 **Chapter Structure Analysis**
        - Analyze chapter lengths and narrative pacing
        - Identify structural patterns in the storytelling
        - Compare chapter complexity across the work
        
        ### 📈 **Vocabulary Progression**
        - Track where new vocabulary is introduced
        - Identify sections with high linguistic novelty
        - Understand the complexity evolution throughout the text
        
        ### 🔍 **Keyword Frequency Analysis**
        - Search for specific terms and themes
        - Visualize keyword usage across chapters
        - Track thematic development throughout the narrative
        
        ### 📋 **Comprehensive Summary**
        - Combined insights from all analytical approaches
        - Key statistics and patterns
        - Entity distribution and network metrics
        
        **Choose an input method from the sidebar to begin your analysis!**
        
        *Supports PDF files, text files, or direct text input*
        """)

if __name__ == "__main__":
    main()