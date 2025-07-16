import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import re
from collections import defaultdict

# Configure page
st.set_page_config(page_title="Silmarillion Network", layout="wide")

class SilmarillionAnalyzer:
    def __init__(self):
        # Core entities from the Silmarillion
        self.entities = {
            'Eru', 'Ilúvatar', 'Melkor', 'Morgoth', 'Manwë', 'Varda', 'Elbereth',
            'Aulë', 'Yavanna', 'Oromë', 'Ulmo', 'Mandos', 'Nienna', 'Tulkas', 'Sauron',
            'Fëanor', 'Fingolfin', 'Finarfin', 'Finwë', 'Maedhros', 'Maglor',
            'Celegorm', 'Caranthir', 'Curufin', 'Amrod', 'Amras', 'Fingon',
            'Turgon', 'Aredhel', 'Galadriel', 'Elrond', 'Elros', 'Gil-galad',
            'Thingol', 'Elwë', 'Melian', 'Lúthien', 'Beren', 'Túrin', 'Tuor',
            'Eärendil', 'Elwing', 'Idril',
            'Valinor', 'Aman', 'Beleriand', 'Middle-earth', 'Angband', 'Thangorodrim',
            'Gondolin', 'Doriath', 'Nargothrond', 'Menegroth', 'Alqualondë',
            'Tirion', 'Númenor', 'Lindon',
            'Silmaril', 'Silmarils', 'Palantír', 'Narsil', 'Ringil', 'Gurthang',
            'Ainur', 'Valar', 'Maiar', 'Eldar', 'Elves', 'Noldor', 'Vanyar',
            'Teleri', 'Sindar', 'Edain', 'Men', 'Dwarves', 'Orcs'
        }
        self.graph = nx.Graph()
        self.text = ""
        self.entity_mentions = defaultdict(int)
    
    def extract_from_pdf(self, pdf_file):
        """Extract text from PDF"""
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            total_pages = len(reader.pages)
            
            progress = st.progress(0)
            for i, page in enumerate(reader.pages):
                try:
                    text += page.extract_text() + "\n"
                    progress.progress((i + 1) / total_pages)
                except:
                    continue
            
            progress.empty()
            return text
        except ImportError:
            st.error("Install PyPDF2: pip install PyPDF2")
            return None
        except Exception as e:
            st.error(f"PDF error: {e}")
            return None
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove common artifacts
        text = re.sub(r'illustration by.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'copyright.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def analyze(self, text):
        """Find entity relationships"""
        self.text = self.clean_text(text)
        sentences = re.split(r'[.!?]+', self.text)
        
        relationships = defaultdict(int)
        self.entity_mentions = defaultdict(int)
        
        # Find entities and co-occurrences
        for sentence in sentences:
            found = []
            for entity in self.entities:
                if entity.lower() in sentence.lower():
                    found.append(entity)
                    self.entity_mentions[entity] += 1
            
            # Create relationships
            for i, e1 in enumerate(found):
                for e2 in found[i+1:]:
                    pair = tuple(sorted([e1, e2]))
                    relationships[pair] += 1
        
        # Build graph
        self.graph = nx.Graph()
        
        # Add nodes
        for entity, count in self.entity_mentions.items():
            if count > 0:
                self.graph.add_node(entity, mentions=count)
        
        # Add edges
        for (e1, e2), strength in relationships.items():
            if strength > 1:
                self.graph.add_edge(e1, e2, weight=strength)
        
        return len(self.graph.nodes), len(self.graph.edges)
    
    def create_plot(self):
        """Create network visualization"""
        if len(self.graph.nodes) == 0:
            return None
        
        pos = nx.spring_layout(self.graph, k=3, iterations=50)
        
        # Node data
        node_x, node_y, node_size, node_text, node_hover = [], [], [], [], []
        node_color = []
        
        # Entity type colors
        major_chars = {'Eru', 'Melkor', 'Morgoth', 'Fëanor', 'Fingolfin', 'Lúthien', 'Beren'}
        places = {'Valinor', 'Beleriand', 'Middle-earth', 'Angband', 'Gondolin', 'Doriath'}
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            mentions = self.graph.nodes[node]['mentions']
            node_size.append(max(15, min(60, mentions * 3 + 15)))
            node_text.append(node)
            node_hover.append(f"{node}<br>Mentions: {mentions}<br>Connections: {self.graph.degree[node]}")
            
            # Color by type
            if node in major_chars:
                node_color.append('#ff6b6b')
            elif node in places:
                node_color.append('#4ecdc4')
            else:
                node_color.append('#95e1d3')
        
        # Edge data
        edge_x, edge_y = [], []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=12, color='white'),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_hover,
            showlegend=False
        ))
        
        fig.update_layout(
            title="Silmarillion Character Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=1000
        )
        
        return fig

# Main app
def main():
    st.title("Silmarillion Network Explorer")
    st.markdown("*Discover character relationships in Tolkien's mythology*")
    
    # Initialize
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SilmarillionAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar
    with st.sidebar:
        st.header("📖 Input")
        
        method = st.radio("Choose input:", ["Upload PDF", "Upload TXT", "Paste Text", "Sample"])
        
        text = None
        
        if method == "Upload PDF":
            file = st.file_uploader("Upload PDF", type=['pdf'])
            if file:
                with st.spinner("Extracting from PDF..."):
                    text = analyzer.extract_from_pdf(file)
                if text:
                    st.success(f"✅ Extracted {len(text):,} characters")
                    
                    # Quality check
                    silmarillion_terms = ['Eru', 'Melkor', 'Fëanor', 'Silmaril', 'Valinor']
                    found_terms = sum(1 for term in silmarillion_terms if term.lower() in text.lower())
                    
                    if found_terms >= 3:
                        st.success(f"✅ Appears to be Silmarillion ({found_terms}/5 key terms)")
                    else:
                        st.warning(f"⚠️ May not be Silmarillion ({found_terms}/5 key terms)")
        
        elif method == "Upload TXT":
            file = st.file_uploader("Upload TXT", type=['txt'])
            if file:
                text = str(file.read(), "utf-8")
                st.success(f"✅ Loaded {len(text):,} characters")
        
        elif method == "Paste Text":
            text = st.text_area("Paste text:", height=200)
            if text:
                st.success(f"✅ {len(text):,} characters")
        
        else:  # Sample
            text = """
            But Morgoth lusted after the Silmarils and the hatred of Fëanor was added 
            to the malice of Melkor. And Morgoth took the Silmarils and set them in 
            his iron crown. Then Fëanor spoke before the Noldor, and swore a terrible 
            oath, he and his seven sons. Among the tales of sorrow and of ruin that 
            come down to us from the darkness of those days there are yet some in which 
            amid weeping there is joy. And of these histories most fair still in the 
            ears of the Elves is the tale of Beren and Lúthien. For Beren was a mortal 
            man, but Lúthien was the daughter of Thingol, a king of Elves upon Middle-earth.
            The Noldor under Fingolfin took the northern lands, and Fingon his son held 
            the realm between the mountains. Turgon built the hidden city of Gondolin in 
            the Encircling Mountains. From Valinor came the light of the Two Trees, but 
            Melkor destroyed them in his malice. In Angband he dwelt with his creatures.
            """
            st.info("📝 Using sample text")
        
        # Analysis controls
        if text:
            st.markdown("---")
            st.header("⚙️ Analysis Controls")
            
            min_mentions = st.slider(
                "Minimum mentions:", 
                min_value=1, max_value=20, value=2,
                help="Filter entities mentioned fewer times"
            )
            
            min_connections = st.slider(
                "Minimum relationship strength:", 
                min_value=1, max_value=10, value=2,
                help="Require stronger co-occurrence"
            )
            
            # Network layout options
            layout_type = st.selectbox(
                "Network layout:",
                ["spring", "circular", "kamada_kawai"],
                help="How to arrange the network nodes"
            )
            
            # Entity focus
            if 'analyzer' in st.session_state and st.session_state.analyzer.entity_mentions:
                focus_entity = st.selectbox(
                    "Focus on entity:",
                    ["None"] + sorted(st.session_state.analyzer.entity_mentions.keys()),
                    help="Center the network on a specific entity"
                )
            else:
                focus_entity = "None"
    
    # Main content
    if text:
        if st.button("🔍 Analyze Network", type="primary"):
            with st.spinner("Finding relationships..."):
                nodes, edges = analyzer.analyze(text)
                st.success(f"✅ Found {nodes} entities, {edges} relationships")
        
        if analyzer.graph.number_of_nodes() > 0:
            # Quick stats dashboard
            st.subheader("Network Overview")
            
            stats = {
                'nodes': len(analyzer.graph.nodes),
                'edges': len(analyzer.graph.edges),
                'density': nx.density(analyzer.graph)
            }
            
            # Calculate additional metrics
            if analyzer.entity_mentions:
                most_mentioned = max(analyzer.entity_mentions.items(), key=lambda x: x[1])
                most_connected = max(dict(analyzer.graph.degree()).items(), key=lambda x: x[1])
                avg_mentions = sum(analyzer.entity_mentions.values()) / len(analyzer.entity_mentions)
                avg_connections = sum(dict(analyzer.graph.degree()).values()) / len(analyzer.graph.nodes)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Entities", 
                    stats['nodes'],
                    help="Unique characters, places, and artifacts found"
                )
            with col2:
                st.metric(
                    "Total Relationships", 
                    stats['edges'],
                    help="Connections between entities"
                )
            with col3:
                st.metric(
                    "Network Density", 
                    f"{stats['density']:.3f}",
                    help="How interconnected the network is (0-1)"
                )
            with col4:
                st.metric(
                    "Avg Connections", 
                    f"{avg_connections:.1f}",
                    help="Average number of connections per entity"
                )
            
            # Key insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Most Mentioned",
                    most_mentioned[0],
                    f"{most_mentioned[1]} times"
                )
            with col2:
                st.metric(
                    "Most Connected", 
                    most_connected[0],
                    f"{most_connected[1]} connections"
                )
            with col3:
                st.metric(
                    "Avg Mentions",
                    f"{avg_mentions:.1f}",
                    help="Average mentions per entity"
                )
            
            # Search functionality
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                search_entity = st.text_input(
                    "🔍 Search for an entity:",
                    placeholder="Type character, place, or artifact name..."
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("Find Connections"):
                    if search_entity and search_entity in analyzer.entity_mentions:
                        # Show entity details
                        st.success(f"Found: {search_entity}")
                        
                        neighbors = list(analyzer.graph.neighbors(search_entity))
                        mentions = analyzer.entity_mentions[search_entity]
                        
                        st.write(f"**{search_entity}** appears {mentions} times and connects to {len(neighbors)} other entities:")
                        
                        if neighbors:
                            # Sort neighbors by connection strength
                            neighbor_strengths = []
                            for neighbor in neighbors:
                                if analyzer.graph.has_edge(search_entity, neighbor):
                                    weight = analyzer.graph[search_entity][neighbor].get('weight', 1)
                                    neighbor_strengths.append((neighbor, weight))
                            
                            neighbor_strengths.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display as columns
                            cols = st.columns(3)
                            for i, (neighbor, strength) in enumerate(neighbor_strengths[:15]):
                                with cols[i % 3]:
                                    st.write(f"• **{neighbor}** ({strength})")
                    elif search_entity:
                        st.error(f"'{search_entity}' not found in the network")
            
            # Main network visualization
            st.markdown("---")
            st.subheader("🕸️ Character Network")
            
            # Apply filters to the visualization
            filtered_graph = analyzer.graph.copy()
            
            # Filter by mentions
            if 'min_mentions' in locals():
                nodes_to_remove = [node for node in filtered_graph.nodes() 
                                 if analyzer.entity_mentions.get(node, 0) < min_mentions]
                filtered_graph.remove_nodes_from(nodes_to_remove)
            
            # Filter by connection strength
            if 'min_connections' in locals():
                edges_to_remove = [(u, v) for u, v, d in filtered_graph.edges(data=True) 
                                 if d.get('weight', 1) < min_connections]
                filtered_graph.remove_edges_from(edges_to_remove)
            
            # Update analyzer's graph temporarily for visualization
            original_graph = analyzer.graph
            analyzer.graph = filtered_graph
            
            fig = analyzer.create_plot()
            if fig:
                # Update title with filter info
                filter_info = f"Showing {len(filtered_graph.nodes)} entities, {len(filtered_graph.edges)} relationships"
                fig.update_layout(title=f"Silmarillion Character Network<br><sub>{filter_info}</sub>")
                st.plotly_chart(fig, use_container_width=True)
            
            # Restore original graph
            analyzer.graph = original_graph
            
            # Additional visualizations
            st.markdown("---")
            st.subheader("Network Analysis")
            
            # Create additional charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Entity type distribution pie chart
                if analyzer.entity_mentions:
                    entity_types = defaultdict(int)
                    major_chars = {'Eru', 'Melkor', 'Morgoth', 'Fëanor', 'Fingolfin', 'Lúthien', 'Beren'}
                    places = {'Valinor', 'Beleriand', 'Middle-earth', 'Angband', 'Gondolin', 'Doriath'}
                    
                    for entity in analyzer.entity_mentions.keys():
                        if entity in major_chars:
                            entity_types['Major Characters'] += 1
                        elif entity in places:
                            entity_types['Places'] += 1
                        elif 'Silmaril' in entity:
                            entity_types['Artifacts'] += 1
                        else:
                            entity_types['Other Entities'] += 1
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(entity_types.keys()),
                        values=list(entity_types.values()),
                        marker_colors=['#ff6b6b', '#4ecdc4', '#ffa07a', '#95e1d3']
                    )])
                    fig_pie.update_layout(
                        title="Entity Type Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Network degree distribution
                if analyzer.graph.nodes:
                    degrees = [analyzer.graph.degree[node] for node in analyzer.graph.nodes()]
                    
                    fig_hist = go.Figure(data=[go.Histogram(
                        x=degrees,
                        nbinsx=15,
                        marker_color='#4ecdc4',
                        opacity=0.7
                    )])
                    fig_hist.update_layout(
                        title="Connection Distribution",
                        xaxis_title="Number of Connections",
                        yaxis_title="Number of Entities",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            # Mention frequency visualization
            st.subheader("Entity Importance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top entities by mentions - bar chart
                if analyzer.entity_mentions:
                    top_mentions = sorted(analyzer.entity_mentions.items(), 
                                        key=lambda x: x[1], reverse=True)[:15]
                    
                    fig_bar = go.Figure(data=[go.Bar(
                        x=[item[1] for item in top_mentions],
                        y=[item[0] for item in top_mentions],
                        orientation='h',
                        marker_color='#ff6b6b',
                        text=[item[1] for item in top_mentions],
                        textposition='auto'
                    )])
                    fig_bar.update_layout(
                        title="Most Mentioned Entities",
                        xaxis_title="Number of Mentions",
                        height=800,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Network centrality analysis
                if analyzer.graph.nodes:
                    # Calculate different centrality measures
                    degree_cent = nx.degree_centrality(analyzer.graph)
                    betweenness_cent = nx.betweenness_centrality(analyzer.graph)
                    
                    # Create comparison chart
                    entities = list(degree_cent.keys())[:15]  # Top 15
                    degree_values = [degree_cent[e] for e in entities]
                    betweenness_values = [betweenness_cent[e] for e in entities]
                    
                    fig_cent = go.Figure()
                    fig_cent.add_trace(go.Scatter(
                        x=degree_values,
                        y=betweenness_values,
                        mode='markers+text',
                        text=entities,
                        textposition='top center',
                        marker=dict(
                            size=[analyzer.entity_mentions.get(e, 1) * 2 for e in entities],
                            color='#95e1d3',
                            opacity=0.7,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Degree Centrality: %{x:.3f}<br>' +
                                    'Betweenness Centrality: %{y:.3f}<br>' +
                                    '<extra></extra>'
                    ))
                    
                    fig_cent.update_layout(
                        title="Network Centrality Analysis",
                        xaxis_title="Degree Centrality",
                        yaxis_title="Betweenness Centrality",
                        height=800
                    )
                    st.plotly_chart(fig_cent, use_container_width=True)
            
            # Connection strength heatmap
            st.subheader("Strongest Relationships")
            
            if analyzer.graph.edges:
                # Get strongest relationships
                edge_weights = []
                for edge in analyzer.graph.edges(data=True):
                    weight = edge[2].get('weight', 1)
                    edge_weights.append((edge[0], edge[1], weight))
                
                # Sort by weight and take top relationships
                top_edges = sorted(edge_weights, key=lambda x: x[2], reverse=True)[:20]
                
                # Create a relationship strength chart
                relationship_data = []
                for e1, e2, weight in top_edges:
                    relationship_data.append({
                        'Relationship': f"{e1} ↔ {e2}",
                        'Strength': weight,
                        'Entity 1': e1,
                        'Entity 2': e2
                    })
                
                df_relationships = pd.DataFrame(relationship_data)
                
                fig_rel = go.Figure(data=[go.Bar(
                    x=df_relationships['Strength'],
                    y=df_relationships['Relationship'],
                    orientation='h',
                    marker_color='#ffa07a',
                    text=df_relationships['Strength'],
                    textposition='auto'
                )])
                
                fig_rel.update_layout(
                    title="Strongest Character Relationships",
                    xaxis_title="Co-occurrence Frequency",
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_rel, use_container_width=True)
            
            # Entity tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Most Mentioned")
                if analyzer.entity_mentions:
                    mentions_data = [
                        {"Entity": entity, "Mentions": count}
                        for entity, count in sorted(analyzer.entity_mentions.items(), 
                                                   key=lambda x: x[1], reverse=True)[:15]
                    ]
                    st.dataframe(pd.DataFrame(mentions_data), use_container_width=True)
            
            with col2:
                st.subheader("🔗 Most Connected")
                if analyzer.graph.nodes:
                    degree_data = [
                        {"Entity": entity, "Connections": degree}
                        for entity, degree in sorted(dict(analyzer.graph.degree()).items(), 
                                                    key=lambda x: x[1], reverse=True)[:15]
                    ]
                    st.dataframe(pd.DataFrame(degree_data), use_container_width=True)
        
        else:
            st.info("👆 Click 'Analyze Network' to start")
    
    else:
        st.markdown("""
        ### Welcome! 
        
        This tool analyzes character relationships in the Silmarillion by:
        - Finding mentions of key characters, places, and artifacts
        - Detecting when entities appear together in sentences
        - Building a network showing their connections
        - Visualizing the relationships interactively
        
        **Choose an input method from the sidebar to start!**
        """)

if __name__ == "__main__":
    main()