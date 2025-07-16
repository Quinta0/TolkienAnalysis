#!/usr/bin/env python3
"""
Simple Silmarillion Network Analyzer
~250 lines of focused code to extract and visualize character relationships
Supports TXT and PDF files
"""

import re
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter
import streamlit as st
import pandas as pd

class SimpleSilmarillionAnalyzer:
    def __init__(self):
        # Core Silmarillion entities (manually curated for accuracy)
        self.entities = {
            # Major Ainur
            'Eru', 'Ilúvatar', 'Melkor', 'Morgoth', 'Manwë', 'Varda', 'Elbereth',
            'Aulë', 'Yavanna', 'Oromë', 'Ulmo', 'Mandos', 'Nienna', 'Tulkas', 'Sauron',
            
            # Noldor
            'Fëanor', 'Fingolfin', 'Finarfin', 'Finwë', 'Maedhros', 'Maglor',
            'Celegorm', 'Caranthir', 'Curufin', 'Amrod', 'Amras', 'Fingon',
            'Turgon', 'Aredhel', 'Galadriel', 'Elrond', 'Elros', 'Gil-galad',
            
            # Sindar & Others
            'Thingol', 'Elwë', 'Melian', 'Lúthien', 'Beren', 'Túrin', 'Tuor',
            'Eärendil', 'Elwing', 'Idril',
            
            # Places
            'Valinor', 'Aman', 'Beleriand', 'Middle-earth', 'Angband', 'Thangorodrim',
            'Gondolin', 'Doriath', 'Nargothrond', 'Menegroth', 'Alqualondë',
            'Tirion', 'Númenor', 'Lindon',
            
            # Artifacts
            'Silmaril', 'Silmarils', 'Palantír', 'Narsil', 'Ringil', 'Gurthang',
            
            # Groups
            'Ainur', 'Valar', 'Maiar', 'Eldar', 'Elves', 'Noldor', 'Vanyar',
            'Teleri', 'Sindar', 'Edain', 'Men', 'Dwarves', 'Orcs'
        }
        
        self.graph = nx.Graph()
        self.text = ""
        self.relationships = []
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            # Read PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Show progress for large PDFs
            total_pages = len(pdf_reader.pages)
            progress_placeholder = st.empty()
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    
                    # Update progress every 10 pages
                    if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                        progress_placeholder.text(f"Extracting PDF... {page_num + 1}/{total_pages} pages")
                        
                except Exception as e:
                    st.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            progress_placeholder.empty()
            return text
            
        except ImportError:
            st.error("PDF support requires PyPDF2. Install it with: pip install PyPDF2")
            return None
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    
    def load_text(self, text):
        """Load and clean text"""
        self.text = text
        
        # Simple cleaning - remove obvious artifacts
        patterns_to_remove = [
            r'illustration by.*',
            r'copyright.*',
            r'isbn.*',
            r'page \d+',
            r'^\d+'
            r'houghton mifflin.*',
            r'all rights reserved.*'
        ]
        

# Streamlit App
def main():
    st.set_page_config(page_title="Silmarillion Network", layout="wide")
    
    st.title("⚔️ Silmarillion Network Explorer")
    st.markdown("*Discover the hidden connections in Tolkien's mythology*")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleSilmarillionAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for input
    with st.sidebar:
        st.header("📖 Load Text")
        
        # Text input options
        input_method = st.radio("Choose input method:", 
                               ["Upload file", "Paste text", "Use sample"])
        
        if input_method == "Upload file":
            uploaded_file = st.file_uploader("Upload Silmarillion text", 
                                           type=['txt', 'pdf'])
            if uploaded_file:
                text = None
                
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                    st.success("✅ Text file loaded")
                
                elif uploaded_file.type == "application/pdf":
                    with st.spinner("Extracting text from PDF..."):
                        text = analyzer.extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        st.success("✅ PDF text extracted")
                        
                        # Show extraction info
                        with st.expander("📄 PDF Extraction Details"):
                            st.write(f"**Total characters extracted:** {len(text):,}")
                            st.write(f"**Estimated pages:** ~{len(text) // 2000}")
                            
                            # Show sample of extracted text
                            sample = text[:500] + "..." if len(text) > 500 else text
                            st.text_area("Sample extracted text:", sample, height=100)
                    else:
                        st.error("❌ Failed to extract text from PDF")
                
                if text:
                    char_count = analyzer.load_text(text)
                    st.info(f"📊 Processing {char_count:,} characters")
                    
                    # Quick quality check
                    silmarillion_terms = ['Eru', 'Melkor', 'Fëanor', 'Silmaril', 'Valinor']
                    found_terms = sum(1 for term in silmarillion_terms if term.lower() in text.lower())
                    
                    if found_terms >= 3:
                        st.success(f"✅ Text appears to be Silmarillion ({found_terms}/5 key terms found)")
                    else:
                        st.warning(f"⚠️ Text may not be Silmarillion ({found_terms}/5 key terms found)")
        
        elif input_method == "Paste text":
            text = st.text_area("Paste Silmarillion text here:", height=200,
                              placeholder="Paste the text content of the Silmarillion here...")
            if text:
                char_count = analyzer.load_text(text)
                st.success(f"✅ Loaded {char_count:,} characters")
        
        else:  # Sample text
            sample = """
            But Morgoth lusted after the Silmarils and the hatred of Fëanor was added 
            to the malice of Melkor. And Morgoth took the Silmarils and set them in 
            his iron crown. Then Fëanor spoke before the Noldor, and swore a terrible 
            oath, he and his seven sons. In that time were made those things that 
            afterwards were most renowned of all the works of the Elves. For Fëanor, 
            being come to his full might, was filled with a new thought. Among the 
            tales of sorrow and of ruin that come down to us from the darkness of 
            those days there are yet some in which amid weeping there is joy. And of 
            these histories most fair still in the ears of the Elves is the tale of 
            Beren and Lúthien. For Beren was a mortal man, but Lúthien was the 
            daughter of Thingol, a king of Elves upon Middle-earth when the world was young.
            And it came to pass that Beren came upon Lúthien suddenly in the woods of 
            Neldoreth, when the trees were green and birds were singing in the spring.
            Then the spell of her beauty fell upon him, and he named her Tinúviel.
            But Melkor dwelt in Angband, and there he made for himself a great fortress
            beneath Thangorodrim. And from thence he sent forth his creatures, Orcs and
            Balrogs and great worms. The Noldor under Fingolfin took the northern lands,
            and Fingon his son held the realm between the mountains. Turgon built the
            hidden city of Gondolin in the Encircling Mountains. But greatest of all the
            works of the Noldor was Nargothrond, the fortress and halls delved in the
            banks of the river Narog.
            """
            char_count = analyzer.load_text(sample)
            st.info(f"📝 Using sample text ({char_count:,} characters)")
        
        # Quick settings
        if analyzer.text:
            st.markdown("---")
            st.subheader("⚙️ Settings")
            
            min_mentions = st.slider("Minimum mentions to include entity:", 
                                    min_value=1, max_value=10, value=2,
                                    help="Filter out entities mentioned fewer times")
            
            min_connections = st.slider("Minimum co-occurrences for relationship:", 
                                      min_value=1, max_value=5, value=2,
                                      help="Require entities to appear together multiple times")
            
            # Store settings in session state
            st.session_state.min_mentions = min_mentions
            st.session_state.min_connections = min_connections
    
    # Main content
    if analyzer.text:
        if st.button("🔍 Analyze Relationships", type="primary"):
            with st.spinner("Finding character relationships..."):
                nodes, edges = analyzer.find_relationships()
                st.success(f"✅ Found {nodes} entities and {edges} relationships")
        
        if analyzer.graph.number_of_nodes() > 0:
            # Statistics
            stats = analyzer.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entities", stats['nodes'])
            with col2:
                st.metric("Connections", stats['edges'])
            with col3:
                st.metric("Most Mentioned", 
                         f"{stats['most_mentioned'][0]} ({stats['most_mentioned'][1]})")
            with col4:
                st.metric("Most Connected", 
                         f"{stats['most_connected'][0]} ({stats['most_connected'][1]})")
            
            # Network visualization
            st.subheader("🕸️ Character Network")
            fig = analyzer.create_network_plot()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Entity list
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Most Mentioned")
                mentions_df = pd.DataFrame([
                    {"Entity": entity, "Mentions": count} 
                    for entity, count in sorted(analyzer.entity_mentions.items(), 
                                              key=lambda x: x[1], reverse=True)[:15]
                ])
                st.dataframe(mentions_df, use_container_width=True)
            
            with col2:
                st.subheader("🔗 Most Connected")
                connections_df = pd.DataFrame([
                    {"Entity": entity, "Connections": degree} 
                    for entity, degree in sorted(dict(analyzer.graph.degree()).items(), 
                                                key=lambda x: x[1], reverse=True)[:15]
                ])
                st.dataframe(connections_df, use_container_width=True)
        
        else:
            st.info("👆 Click 'Analyze Relationships' to start the analysis")
    
    else:
        st.info("👈 Load some text from the sidebar to get started")
        
        # Show what the tool does
        st.markdown("""
        ### What this tool does:
        
        1. **Extracts key entities** from Silmarillion text (characters, places, artifacts)
        2. **Finds relationships** by analyzing which entities appear together
        3. **Visualizes the network** showing connections between entities
        4. **Provides statistics** about the most important characters and relationships
        
        ### Perfect for:
        - Understanding character relationships
        - Discovering hidden connections
        - Exploring Tolkien's world structure
        - Academic analysis of the text
        """)

if __name__ == "__main__":
    main()
    for pattern in patterns_to_remove:
        self.text = re.sub(pattern, '', self.text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean whitespace
        self.text = re.sub(r'\s+', ' ', self.text)
        return len(self.text)
    
    def find_relationships(self):
        """Find relationships between entities using simple co-occurrence"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', self.text)
        
        relationships = defaultdict(int)
        entity_mentions = defaultdict(int)
        
        for sentence in sentences:
            # Find entities in this sentence
            found_entities = []
            for entity in self.entities:
                if entity.lower() in sentence.lower():
                    found_entities.append(entity)
                    entity_mentions[entity] += 1
            
            # Create relationships between entities in same sentence
            for i, entity1 in enumerate(found_entities):
                for entity2 in found_entities[i+1:]:
                    # Create consistent pair ordering
                    pair = tuple(sorted([entity1, entity2]))
                    relationships[pair] += 1
        
        self.relationships = relationships
        self.entity_mentions = entity_mentions
        
        # Build network
        self.graph = nx.Graph()
        
        # Add nodes with mention counts
        for entity, count in entity_mentions.items():
            if count > 0:  # Only entities that appear in text
                self.graph.add_node(entity, mentions=count)
        
        # Add edges with relationship strength
        for (entity1, entity2), strength in relationships.items():
            if strength > 1:  # Only meaningful co-occurrences
                self.graph.add_edge(entity1, entity2, weight=strength)
        
        return len(self.graph.nodes), len(self.graph.edges)
    
    def get_entity_type(self, entity):
        """Classify entity type for coloring"""
        major_chars = {'Eru', 'Ilúvatar', 'Melkor', 'Morgoth', 'Manwë', 'Varda', 'Fëanor', 'Fingolfin', 'Lúthien', 'Beren', 'Sauron'}
        places = {'Valinor', 'Beleriand', 'Middle-earth', 'Angband', 'Gondolin', 'Doriath', 'Númenor'}
        artifacts = {'Silmaril', 'Silmarils', 'Palantír', 'Narsil'}
        
        if entity in major_chars:
            return 'major_character'
        elif entity in places:
            return 'place'
        elif entity in artifacts:
            return 'artifact'
        else:
            return 'other'
    
    def create_network_plot(self):
        """Create interactive network visualization"""
        if len(self.graph.nodes) == 0:
            return None
        
        # Layout
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Colors for different entity types
        colors = {
            'major_character': '#ff6b6b',
            'place': '#4ecdc4',
            'artifact': '#ffa07a',
            'other': '#95e1d3'
        }
        
        # Prepare node data
        node_x, node_y, node_color, node_size, node_text, node_hover = [], [], [], [], [], []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            entity_type = self.get_entity_type(node)
            node_color.append(colors[entity_type])
            
            mentions = self.graph.nodes[node]['mentions']
            node_size.append(max(10, min(50, mentions * 2 + 10)))
            
            node_text.append(node)
            node_hover.append(f"<b>{node}</b><br>Mentions: {mentions}<br>Connections: {self.graph.degree[node]}")
        
        # Prepare edge data
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_hover,
            showlegend=False
        ))
        
        fig.update_layout(
            title="Silmarillion Character Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        return fig
    
    def get_stats(self):
        """Get basic network statistics"""
        if len(self.graph.nodes) == 0:
            return {}
        
        return {
            'nodes': len(self.graph.nodes),
            'edges': len(self.graph.edges),
            'density': nx.density(self.graph),
            'most_mentioned': max(self.entity_mentions.items(), key=lambda x: x[1]),
            'most_connected': max(dict(self.graph.degree()).items(), key=lambda x: x[1])
        }

# Streamlit App
def main():
    st.set_page_config(page_title="Silmarillion Network", layout="wide")
    
    st.title("⚔️ Silmarillion Network Explorer")
    st.markdown("*Discover the hidden connections in Tolkien's mythology*")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleSilmarillionAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for input
    with st.sidebar:
        st.header("📖 Load Text")
        
        # Text input options
        input_method = st.radio("Choose input method:", 
                               ["Upload file", "Paste text", "Use sample"])
        
        if input_method == "Upload file":
            uploaded_file = st.file_uploader("Upload Silmarillion text", 
                                           type=['txt', 'pdf'])
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("PDF support coming soon. Use .txt for now.")
                    text = None
                
                if text:
                    char_count = analyzer.load_text(text)
                    st.success(f"✅ Loaded {char_count:,} characters")
        
        elif input_method == "Paste text":
            text = st.text_area("Paste Silmarillion text here:", height=200)
            if text:
                char_count = analyzer.load_text(text)
                st.success(f"✅ Loaded {char_count:,} characters")
        
        else:  # Sample text
            sample = """
            But Morgoth lusted after the Silmarils and the hatred of Fëanor was added 
            to the malice of Melkor. And Morgoth took the Silmarils and set them in 
            his iron crown. Then Fëanor spoke before the Noldor, and swore a terrible 
            oath, he and his seven sons. In that time were made those things that 
            afterwards were most renowned of all the works of the Elves. For Fëanor, 
            being come to his full might, was filled with a new thought. Among the 
            tales of sorrow and of ruin that come down to us from the darkness of 
            those days there are yet some in which amid weeping there is joy. And of 
            these histories most fair still in the ears of the Elves is the tale of 
            Beren and Lúthien. For Beren was a mortal man, but Lúthien was the 
            daughter of Thingol, a king of Elves upon Middle-earth.
            """
            char_count = analyzer.load_text(sample)
            st.info(f"📝 Using sample text ({char_count:,} characters)")
    
    # Main content
    if analyzer.text:
        if st.button("🔍 Analyze Relationships", type="primary"):
            with st.spinner("Finding character relationships..."):
                nodes, edges = analyzer.find_relationships()
                st.success(f"✅ Found {nodes} entities and {edges} relationships")
        
        if analyzer.graph.number_of_nodes() > 0:
            # Statistics
            stats = analyzer.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entities", stats['nodes'])
            with col2:
                st.metric("Connections", stats['edges'])
            with col3:
                st.metric("Most Mentioned", 
                         f"{stats['most_mentioned'][0]} ({stats['most_mentioned'][1]})")
            with col4:
                st.metric("Most Connected", 
                         f"{stats['most_connected'][0]} ({stats['most_connected'][1]})")
            
            # Network visualization
            st.subheader("🕸️ Character Network")
            fig = analyzer.create_network_plot()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Entity list
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Most Mentioned")
                mentions_df = pd.DataFrame([
                    {"Entity": entity, "Mentions": count} 
                    for entity, count in sorted(analyzer.entity_mentions.items(), 
                                              key=lambda x: x[1], reverse=True)[:15]
                ])
                st.dataframe(mentions_df, use_container_width=True)
            
            with col2:
                st.subheader("🔗 Most Connected")
                connections_df = pd.DataFrame([
                    {"Entity": entity, "Connections": degree} 
                    for entity, degree in sorted(dict(analyzer.graph.degree()).items(), 
                                                key=lambda x: x[1], reverse=True)[:15]
                ])
                st.dataframe(connections_df, use_container_width=True)
        
        else:
            st.info("👆 Click 'Analyze Relationships' to start the analysis")
    
    else:
        st.info("👈 Load some text from the sidebar to get started")
        
        # Show what the tool does
        st.markdown("""
        ### What this tool does:
        
        1. **Extracts key entities** from Silmarillion text (characters, places, artifacts)
        2. **Finds relationships** by analyzing which entities appear together
        3. **Visualizes the network** showing connections between entities
        4. **Provides statistics** about the most important characters and relationships
        
        ### Perfect for:
        - Understanding character relationships
        - Discovering hidden connections
        - Exploring Tolkien's world structure
        - Academic analysis of the text
        """)

if __name__ == "__main__":
    main()