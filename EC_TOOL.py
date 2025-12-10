import streamlit as st
import pandas as pd
import os
import re
import time 

# --- Configuration & Global Paths ---
st.set_page_config(
    page_title="Logic Processor & EC Tool", 
    layout="wide", 
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

LOCAL_DIR = 'data_cache'
FILE_PATH = os.path.join(LOCAL_DIR, 'processed_ec_data.csv')
NOM_CF_COL = 'Nom CF /\nNom CO PLM (CF_CO)' 

# --- Helper Functions ---

def strict_clean_input(text):
    """
    Applies strict filtering rules:
    1. Starts processing from the first '('.
    2. Extracts codes (XXX_NN), brackets, commas, and logic operators.
    3. Converts ',' -> 'OR'.
    4. Normalizes to uppercase AND/OR.
    """
    # 1. Start from the first open bracket '('
    start_index = text.find('(')
    if start_index == -1:
        return ""
    
    # Slice the text to ignore "garbage" before the first bracket
    content_to_process = text[start_index:]
    
    # 2. Define Regex Patterns
    # Codes: 3 alphanumeric chars + '_' + 2 digits (e.g., ABC_12, k1k_00)
    # Operators: (, ), , (comma), AND, OR (case insensitive)
    pattern = r"([a-zA-Z0-9]{3}_\d{2}|[()]|,|(?i:\b(?:and|or)\b))"
    
    matches = re.findall(pattern, content_to_process)
    
    cleaned_tokens = []
    for token in matches:
        token_lower = token.lower()
        
        # 3. Convert ',' -> 'OR'
        if token == ',':
            cleaned_tokens.append('OR')
        
        # Normalize Logic operators
        elif token_lower == 'or':
            cleaned_tokens.append('OR')
        elif token_lower == 'and':
            cleaned_tokens.append('AND')
        
        # Keep Brackets
        elif token in ('(', ')'):
            cleaned_tokens.append(token)
            
        # Keep Codes (Normalize to Uppercase)
        else:
            cleaned_tokens.append(token.upper())

    # Join with spaces
    cleaned_string = ' '.join(cleaned_tokens)
    
    # Cosmetic cleanup: fix spaces around brackets
    cleaned_string = cleaned_string.replace('( ', '(').replace(' )', ')')
    
    return cleaned_string.strip()

def generate_combinations(logic_string):
    if not logic_string: return ""
    
    # --- SAFETY CHECK: Detect Unbalanced Parentheses ---
    open_count = logic_string.count('(')
    close_count = logic_string.count(')')
    
    if open_count != close_count:
        return f"⚠️ Error: Unbalanced Parentheses. Opened: {open_count}, Closed: {close_count}. Please check your input."
    # ---------------------------------------------------

    try:
        # Handle outer parentheses stripping if the whole string is enclosed
        if logic_string.startswith('(') and logic_string.endswith(')'):
            depth = 0
            is_enclosed = True
            for i, char in enumerate(logic_string):
                if char == '(': depth += 1
                elif char == ')': depth -= 1
                if depth == 0 and i < len(logic_string) - 1:
                    is_enclosed = False
                    break
            if is_enclosed: logic_string = logic_string[1:-1]

        or_groups = []
        depth = 0
        # Pad parentheses for splitting
        tokens = logic_string.replace('(', ' ( ').replace(')', ' ) ').split()
        temp_tokens = []
        
        for token in tokens:
            if token == '(':
                depth += 1
                temp_tokens.append(token)
            elif token == ')':
                depth -= 1
                temp_tokens.append(token)
            elif token == 'OR' and depth == 0:
                or_groups.append(" ".join(temp_tokens))
                temp_tokens = []
            else:
                temp_tokens.append(token)
        if temp_tokens: or_groups.append(" ".join(temp_tokens))

        final_lines = []
        for group in or_groups:
            # STRICT REGEX: Matches codes like ABC_12
            codes = re.findall(r'[A-Z0-9]{3}_\d{2}', group, re.IGNORECASE)
            if codes:
                codes = sorted(list(set(codes))) 
                # Normalize to Upper for the output combination
                codes = [c.upper() for c in codes]
                combination_line = "|".join(codes)
                final_lines.append(combination_line)
        return "\n".join(final_lines)
    except Exception as e:
        return f"Error: {e}"

def get_validation_data(combinations_str, results_list):
    """
    Validates combinations and returns detailed data for display and filtering.
    """
    if not combinations_str or not results_list:
        return []

    # Map available codes from Excel results
    code_data = {item['matched_word'].upper(): item for item in results_list}
    validation_data = []
    
    # If the combination string is actually an error message, don't try to split it
    if combinations_str.startswith("⚠️") or combinations_str.startswith("Error"):
        return []

    for line in combinations_str.split('\n'):
        if not line.strip(): continue
        
        codes = line.split('|')
        is_dpeo = True
        is_cdpo = True
        
        for code in codes:
            # Check against the mapped data
            data = code_data.get(code.upper(), {})
            if str(data.get('DPEO', 'N')).strip().upper() != 'Y':
                is_dpeo = False
            if str(data.get('CDPO', 'N')).strip().upper() != 'Y':
                is_cdpo = False
        
        # Determine status
        suffix = ""
        is_valid_for_final = False
        
        if is_dpeo and is_cdpo:
            suffix = " --> DPEO, CDPO"
            is_valid_for_final = True
        elif is_dpeo:
            suffix = " --> DPEO"
            is_valid_for_final = True 
        elif is_cdpo:
            suffix = " --> CDPO"
            is_valid_for_final = True
        else:
            suffix = " --> NOT APPLICABLE to NEA"
            is_valid_for_final = False
            
        validation_data.append({
            "raw_line": line,
            "display_text": f"{line}{suffix}",
            "is_valid": is_valid_for_final
        })
        
    return validation_data

def add_new_request():
    # Sync current widget values to the list before appending
    for i in range(len(st.session_state.request_fields)):
        key = f"request_{i}"
        if key in st.session_state:
            st.session_state.request_fields[i] = st.session_state[key]
            
    st.session_state.request_fields.append('') 
    st.session_state['processed_requests'] = None 
    st.session_state['comparison_results'] = None

def process_requests():
    # --- Interactive Toasts ---
    msg = st.toast('Analyzing inputs...', icon='⏳')
    time.sleep(0.3)
    
    input_list = st.session_state.request_fields
    df = st.session_state['extracted_data']
    
    if df.empty:
        st.session_state.comparison_results = {'error': 'Extraction data is empty. Please upload the file first.'}
        return

    msg.toast('Matching against database...', icon='🔍')
    
    # Prepare comparison column
    df['__COMPARE__'] = df[NOM_CF_COL].astype(str).str.strip().str.lower().apply(lambda x: re.sub(r'[^\w]', '', x))
    
    all_raw_input_text = " ".join(input_list)
    
    # STRICT REGEX: Only extract codes matching the XXX_NN pattern for lookup
    all_codes_in_input = re.findall(r'([a-zA-Z0-9]{3}_\d{2})', all_raw_input_text)
    
    matched_data = {} 
    
    for code in set(all_codes_in_input): 
        clean_code = re.sub(r'[^\w]', '', code).lower()
        if clean_code:
            matched_rows = df[df['__COMPARE__'] == clean_code]
            if not matched_rows.empty:
                for _, row in matched_rows.iterrows():
                    match_key = (code.upper(), row.get('Replacement codes for functional codification', 'N/A'))
                    if match_key not in matched_data:
                        matched_data[match_key] = {
                            'matched_word': code.upper(), 
                            'functional_codification': row.get('Replacement codes for functional codification', 'N/A'),
                            'NEA': row.get('Used before NEA', 'N/A'),
                            'DPEO': row.get('Used from DPEO', 'N/A'),
                            'CDPO': row.get('Used from CDPO', 'N/A'),
                        }
    
    st.session_state.comparison_results = list(matched_data.values())
    if not st.session_state.comparison_results:
        st.session_state.comparison_results = {'info': 'No matching codes found in the EC Referential.'}
    
    st.session_state.processed_requests = input_list 
    msg.toast("Processing complete!", icon='✅')

def initialize_session():
    if 'request_fields' not in st.session_state:
        st.session_state['request_fields'] = [''] 
        st.session_state['processed_requests'] = None
        st.session_state['comparison_results'] = None

    if 'extracted_data' not in st.session_state:
        os.makedirs(LOCAL_DIR, exist_ok=True) 
        if os.path.exists(FILE_PATH):
            try:
                df_initial = pd.read_csv(FILE_PATH)
                st.session_state['extracted_data'] = df_initial
                st.toast("Loaded previous data from local cache.", icon='💾')
            except Exception:
                st.session_state['extracted_data'] = pd.DataFrame()
        else:
            st.session_state['extracted_data'] = pd.DataFrame()

def extract_ec_data_callback():
    file = st.session_state.EC_Referential
    if file is not None:
        try:
            df = pd.read_excel(
                file,
                sheet_name='Liste EC',
                header=1,
                usecols=['EC name /\nDesignationFR CF PLM', 'Values', 
                         NOM_CF_COL, 
                         'Replacement codes for functional codification', 
                         'Used before NEA', 'Used from DPEO', 
                         'Used from CDPO'],
                engine='openpyxl'
            )
            st.session_state['extracted_data'] = df
            os.makedirs(LOCAL_DIR, exist_ok=True) 
            df.to_csv(FILE_PATH, index=False)
            st.toast("✅ Data extracted and saved locally!", icon='💾')
        except Exception as e:
            st.session_state['extracted_data'] = pd.DataFrame() 
            st.error(f"Error reading file/sheet during processing: {e}")

# --- Initialization ---
initialize_session()

# --- MODERN UI STYLING (Soft UI) ---
st.markdown("""
<style>
    /* Global Settings & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp { 
        background-color: #f1f5f9; /* Slate-100 */
    }
    
    /* Premium Cards */
    .css-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #0f172a; /* Slate-900 */
        font-weight: 700;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #ffffff; 
        border: 1px solid #cbd5e1; 
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Consolas', monospace; 
        font-size: 13px;
        color: #334155;
    }
    .stTextArea textarea:focus { 
        border-color: #3b82f6; /* Blue-500 */
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); 
    }

    /* Primary Button (Process) */
    div.stButton > button[kind="primary"] {
        background-color: #2563eb; /* Blue-600 */
        color: white; 
        border-radius: 8px; 
        font-weight: 600; 
        border: none; 
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button[kind="primary"]:hover { 
        background-color: #1d4ed8; /* Blue-700 */
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }

    /* Add Button */
    button[key="add_button"] {
        background-color: #10b981; /* Emerald-500 */
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        padding: 0;
        line-height: 1;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #0f172a;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #2563eb;
        border-bottom: 2px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Data Upload")
    st.file_uploader(
        "Upload your EC Referential File (.xlsm)", 
        type=["xlsm", "xlsx"],
        key="EC_Referential",
        on_change=extract_ec_data_callback
    )
    st.markdown("---")
    if not st.session_state['extracted_data'].empty:
        st.success(f"✅ File Loaded: {len(st.session_state['extracted_data'])} rows")
    else:
        st.info("ℹ️ Awaiting File Upload")

# --- HEADER & STATUS BAR ---
# Top-level status metrics to give immediate context
row_header = st.container()
with row_header:
    col_title, col_m1, col_m2, col_m3 = st.columns([0.4, 0.2, 0.2, 0.2])
    
    with col_title:
        st.title("🚀 Logic Processor")
        st.markdown("*Advanced EC Validation Tool*")
        
    # Metrics
    if not st.session_state['extracted_data'].empty:
        df = st.session_state['extracted_data']
        col_m1.metric("📚 File Status", "Loaded", f"{len(df)} rows")
    else:
        col_m1.metric("📚 File Status", "Waiting", "Upload .xlsm", delta_color="off")

    active_reqs = len([r for r in st.session_state.request_fields if r.strip()])
    col_m2.metric("📝 Active Requests", active_reqs)
    
    # Check match count if processed
    if st.session_state.comparison_results and isinstance(st.session_state.comparison_results, list):
        col_m3.metric("🔍 Matches Found", len(st.session_state.comparison_results))
    else:
        col_m3.metric("🔍 Matches Found", "-", delta_color="off")

st.markdown("---")

# --- SPLIT LAYOUT: COMMAND CENTER ---
col_left, col_right = st.columns([0.35, 0.65], gap="large")

# --- LEFT COLUMN: INPUTS (Control Panel) ---
with col_left:
    st.markdown("### 🛠️ Input Console")
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.caption("Define logic strings to validate against the referenced EC data.")
    
    for i in range(len(st.session_state.request_fields)):
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            st.session_state.request_fields[i] = st.text_area(
                label=f"Request #{i + 1}",
                value=st.session_state.request_fields[i],
                height=100,
                key=f"request_{i}",
                placeholder="HP-0000873 [ ... ] AND ( ... )",
                label_visibility="collapsed" if i > 0 else "visible"
            )
        # Show Add button only on the last item for flow
        if i == len(st.session_state.request_fields) - 1:
            with c2:
                # Spacer to align button
                st.write("") 
                st.write("")
                st.button("＋", on_click=add_new_request, key="add_button", help="Add another request")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process Button (Sticky-ish relative to card)
    st.button("⚡ Run Analysis", on_click=process_requests, key="process_button", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Reference Data "Mini-View" Option could go here if needed, but keeping it simple for now
    with st.expander("📌 Quick Stats"):
        if not st.session_state['extracted_data'].empty:
            st.write(f"Columns: {list(st.session_state['extracted_data'].columns)}")
        else:
            st.write("No data loaded.")


# --- RIGHT COLUMN: RESULTS DECK ---
with col_right:
    # 1. EMPTY STATE (When no processing has happened)
    if st.session_state.processed_requests is None:
        st.markdown('<div class="css-card" style="text-align: center; padding: 4rem;">', unsafe_allow_html=True)
        st.markdown("# 👋 Ready to Start")
        st.markdown("Enter your logic strings in the panel on the left and click **Run Analysis**.")
        st.markdown("Results will appear here instantly.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show Data Preview if file is loaded but no request processed
        if not st.session_state['extracted_data'].empty:
             st.subheader("📚 Reference Data Preview")
             st.dataframe(st.session_state['extracted_data'].head(50), use_container_width=True)

    # 2. RESULTS STATE
    else:
        st.markdown("### 📊 Analysis Results")
        
        # Error/Info Handling
        results_list = st.session_state.comparison_results
        if isinstance(results_list, dict):
            if 'error' in results_list: 
                st.error(results_list['error'])
            elif 'info' in results_list: 
                st.info(results_list['info'])
        
        # Valid Results Tabs
        else:
            valid_requests = []
            for idx, req in enumerate(st.session_state.processed_requests):
                if req.strip() != "":
                    valid_requests.append((idx + 1, req))

            if valid_requests:
                # Use Tabs for multiple requests
                tabs = st.tabs([f"Request #{n}" for n, _ in valid_requests])
                
                for tab_obj, (req_idx, raw_input) in zip(tabs, valid_requests):
                    with tab_obj:
                        st.markdown('<div class="css-card">', unsafe_allow_html=True)
                        
                        # --- Processing Logic ---
                        cleaned_output = strict_clean_input(raw_input)
                        generated_combinations = generate_combinations(cleaned_output)
                        validation_data = get_validation_data(generated_combinations, results_list)

                        # Top Summary for this Request
                        c_summ1, c_summ2 = st.columns(2)
                        with c_summ1:
                            st.caption("Cleaned Logic")
                            st.code(cleaned_output if cleaned_output else "No logic found", language="text")
                        with c_summ2:
                            st.caption("Generated Combinations")
                            if generated_combinations.startswith("⚠️"):
                                st.error(generated_combinations)
                            else:
                                st.code(generated_combinations, language="text")

                        st.markdown("---")
                        
                        # Validation Badges
                        st.markdown("**✅ Combination Validation**")
                        if validation_data:
                            html_badges = '<div style="font-family: monospace; white-space: pre; line-height: 1.6; display: flex; flex-direction: column; gap: 4px;">'
                            for item in validation_data:
                                bg_color = "#ecfdf5" if item['is_valid'] else "#fef2f2" # Emerald-50 / Rose-50
                                border_color = "#10b981" if item['is_valid'] else "#ef4444" # Emerald-500 / Rose-500
                                text_color = "#065f46" if item['is_valid'] else "#991b1b"
                                icon = "✅" if item['is_valid'] else "⛔"
                                
                                html_badges += (
                                    f'<div style="background-color: {bg_color}; border-left: 4px solid {border_color}; '
                                    f'padding: 8px 12px; color: {text_color}; border-radius: 4px; font-size: 0.9rem;">'
                                    f'<b>{icon}</b> {item["display_text"]}</div>'
                                )
                            html_badges += '</div>'
                            st.markdown(html_badges, unsafe_allow_html=True)
                        else:
                            st.info("No combinations to validate.")

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Final Download
                        final_lines = [item['raw_line'] for item in validation_data if item['is_valid']]
                        if final_lines:
                            final_text = "\n".join(final_lines)
                            st.download_button(
                                label="📥 Download Valid Combinations",
                                data=final_text,
                                file_name=f"request_{req_idx}_final.txt",
                                mime="text/plain",
                                help="Download only the valid DPEO/CDPO combinations"
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Matched Codes Details (Accordion style to save space)
                        with st.expander("🔎 View Matched Codes Details", expanded=False):
                            codes_in_this_input = {code.upper() for code in re.findall(r'([a-zA-Z0-9]{3}_\d{2})', raw_input)}
                            df_group = pd.DataFrame([
                                res for res in results_list 
                                if res['matched_word'].upper() in codes_in_this_input
                            ]).drop_duplicates(subset=['matched_word'])

                            if not df_group.empty:
                                df_visual = df_group.copy()
                                for col in ['NEA', 'DPEO', 'CDPO']:
                                    df_visual[col] = df_visual[col].astype(str).str.strip().str.upper() == 'Y'

                                st.dataframe(
                                    df_visual,
                                    use_container_width=True,
                                    column_config={
                                        "matched_word": st.column_config.TextColumn("Code"),
                                        "functional_codification": st.column_config.TextColumn("Replacement"),
                                        "NEA": st.column_config.CheckboxColumn("NEA"),
                                        "DPEO": st.column_config.CheckboxColumn("DPEO"),
                                        "CDPO": st.column_config.CheckboxColumn("CDPO"),
                                    },
                                    hide_index=True
                                )
                            else:
                                st.warning("No reference codes found in this request.")
            else:
                st.info("No input text detected.")
