import streamlit as st
import pandas as pd
import os
import re
import time 

# --- Configuration & Global Paths ---
st.set_page_config(
    page_title="My Test Tool", 
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
        # If no bracket is found, return empty or handle as error depending on preference
        # For now, we return empty string to indicate no valid logic found
        return ""
    
    # Slice the text to ignore "garbage" before the first bracket
    content_to_process = text[start_index:]
    
    # 2. Define Regex Patterns based on roles
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
            
        # Keep Codes (Normalize to Uppercase for consistency with Excel matching)
        else:
            cleaned_tokens.append(token.upper())

    # Join with spaces
    cleaned_string = ' '.join(cleaned_tokens)
    
    # Cosmetic cleanup: fix spaces around brackets
    cleaned_string = cleaned_string.replace('( ', '(').replace(' )', ')')
    
    return cleaned_string.strip()

def generate_combinations(logic_string):
    if not logic_string: return ""
    try:
        # Handle outer parentheses stripping if the whole string is enclosed
        # (Simple check to match opening/closing)
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
            # UPDATE: Use the same strict regex for consistency (XXX_NN)
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
    
    for line in combinations_str.split('\n'):
        if not line.strip(): continue
        
        codes = line.split('|')
        is_dpeo = True
        is_cdpo = True
        
        for code in codes:
            # Check against the mapped data
            data = code_data.get(code.upper(), {})
            # Logic: If ANY code in the line is NOT Y, the whole line fails that check
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
    st.session_state.request_fields.append('') 
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
    
    # Prepare comparison column (strip non-alphanumeric and lower)
    df['__COMPARE__'] = df[NOM_CF_COL].astype(str).str.strip().str.lower().apply(lambda x: re.sub(r'[^\w]', '', x))
    
    all_raw_input_text = " ".join(input_list)
    
    # UPDATE: Use the strict regex here as well to only look up valid codes
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
    /* Global Background */
    .stApp { background-color: #f8f9fa; }
    
    /* Card Style */
    .css-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease;
    }
    .css-card:hover { border-color: #008080; }

    /* Text Area */
    .stTextArea textarea {
        background-color: #ffffff; 
        border: 1px solid #d1d5db; 
        border-radius: 8px;
        font-family: 'Consolas', 'Courier New', monospace; 
        font-size: 14px;
    }
    .stTextArea textarea:focus { border-color: #008080; box-shadow: 0 0 0 1px #008080; }

    /* Buttons */
    .stButton > button {
        background-color: #008080; color: white; border-radius: 8px; 
        font-weight: 600; border: none; padding: 0.5rem 1rem;
    }
    .stButton > button:hover { background-color: #006666; }

    /* Add Button */
    div[data-testid="stVerticalBlock"] > div > div:nth-child(2) > div > button {
        margin-top: 32px; background-color: #4CAF50; width: 42px; height: 42px;
        border-radius: 50%; font-size: 1.5rem; line-height: 1; padding: 0;
    }

    /* Sidebar */
    .stFileUploader { 
        border: 2px dashed #008080; border-radius: 10px; background-color: #f0fdfa; padding: 10px;
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

# --- HEADER ---
st.title("🚀 Logic Processor & EC Tool")

# --- Main Layout ---
tab1, tab2 = st.tabs(["**Dynamic Inputs**", "**Reference Data**"])

# --- TAB 1: Inputs and Processing ---
with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📝 Request Definitions")
    
    for i in range(len(st.session_state.request_fields)):
        col_req, col_add = st.columns([0.92, 0.08])
        widget_label = f"REQUEST #{i + 1}"
        with col_req:
            st.session_state.request_fields[i] = st.text_area(
                label=widget_label, value=st.session_state.request_fields[i],
                height=80, key=f"request_{i}", placeholder="Example: HP-0000873[...]"
            )
        if i == len(st.session_state.request_fields) - 1:
            with col_add:
                st.button("+", on_click=add_new_request, key="add_button", help="Add another request field")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⚡ Process Information", on_click=process_requests, key="process_button", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Results Display with TABS ---
    if st.session_state.processed_requests is not None and not st.session_state['extracted_data'].empty:
        
        # Summary Metrics
        results_list = st.session_state.comparison_results
        if isinstance(results_list, list):
            valid_req_count = len([r for r in st.session_state.processed_requests if r.strip() != ''])
            total_matches = len(results_list)
            
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Requests Processed", valid_req_count)
            m2.metric("Unique Codes Found", total_matches)
            m3.metric("Status", "Complete", delta="Success", delta_color="normal")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("📊 Processing Results")
        
        if isinstance(results_list, dict):
            if 'error' in results_list: st.error(results_list['error'])
            elif 'info' in results_list: st.info(results_list['info'])
        else:
            valid_requests = []
            for idx, req in enumerate(st.session_state.processed_requests):
                if req.strip() != "":
                    valid_requests.append((idx + 1, req))

            if valid_requests:
                result_tabs = st.tabs([f"Request #{idx}" for idx, _ in valid_requests])

                for tab_obj, (req_idx, raw_input) in zip(result_tabs, valid_requests):
                    with tab_obj:
                        st.markdown(f"#### Analysis for Request #{req_idx}")
                        col_out, col_comb = st.columns(2)
                        
                        cleaned_output = strict_clean_input(raw_input)
                        generated_combinations = generate_combinations(cleaned_output)
                        validation_data = get_validation_data(generated_combinations, results_list)

                        with col_out:
                            st.markdown("**Cleaned Logic Output:**")
                            st.info(cleaned_output)
                        
                        with col_comb:
                            st.markdown("**Generated Combinations:**")
                            st.code(generated_combinations, language="text")

                        # --- 1. Display Combination Validation (Badge Style) ---
                        st.markdown("**Combination Validation:**")
                        if validation_data:
                            html_badges = '<div style="font-family: monospace; white-space: pre; line-height: 1.6;">'
                            for item in validation_data:
                                bg_color = "#e6fffa" if item['is_valid'] else "#fff5f5"
                                border_color = "#009933" if item['is_valid'] else "#cc0000"
                                text_color = "#006622" if item['is_valid'] else "#990000"
                                
                                html_badges += (
                                    f'<div style="background-color: {bg_color}; border-left: 4px solid {border_color}; '
                                    f'padding: 4px 10px; margin-bottom: 2px; color: {text_color}; border-radius: 4px;">'
                                    f'{item["display_text"]}</div>'
                                )
                            html_badges += '</div>'
                            st.markdown(html_badges, unsafe_allow_html=True)
                        else:
                            st.info("No combinations to validate.")

                        # --- 2. Display Final Combination (Filtered) ---
                        st.markdown("**Final Combination:**")
                        final_lines = [
                            item['raw_line'] 
                            for item in validation_data 
                            if item['is_valid']
                        ]
                        
                        col_text, col_dl = st.columns([0.85, 0.15])
                        if final_lines:
                            final_text = "\n".join(final_lines)
                            with col_text:
                                st.code(final_text, language="text")
                            with col_dl:
                                st.download_button(
                                    label="📥 Download",
                                    data=final_text,
                                    file_name=f"request_{req_idx}_final.txt",
                                    mime="text/plain"
                                )
                        else:
                            st.info("No valid combinations found (All lines were NOT APPLICABLE).")

                        # --- Matched Codes Table ---
                        # UPDATE: Ensure regex here also matches the strict 2-digit rule
                        codes_in_this_input = {code.upper() for code in re.findall(r'([a-zA-Z0-9]{3}_\d{2})', raw_input)}
                        df_group = pd.DataFrame([
                            res for res in results_list 
                            if res['matched_word'].upper() in codes_in_this_input
                        ]).drop_duplicates(subset=['matched_word'])

                        if not df_group.empty:
                            st.markdown("**Matched Codes Detail:**")
                            df_visual = df_group.copy()
                            for col in ['NEA', 'DPEO', 'CDPO']:
                                df_visual[col] = df_visual[col].astype(str).str.strip().str.upper() == 'Y'

                            st.dataframe(
                                df_visual,
                                width="stretch",
                                column_config={
                                    "matched_word": st.column_config.TextColumn("Nom CF / Nom CO PLM (CF_CO)"),
                                    "functional_codification": st.column_config.TextColumn("Replacement codes"),
                                    "NEA": st.column_config.CheckboxColumn("NEA", width="small"),
                                    "DPEO": st.column_config.CheckboxColumn("DPEO", width="small"),
                                    "CDPO": st.column_config.CheckboxColumn("CDPO", width="small"),
                                },
                                hide_index=True
                            )
                        else:
                            st.warning(f"No valid codes found matching the reference data for Request #{req_idx}")
                            
            else:
                st.info("No text entered in any request fields.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.comparison_results is not None:
        if isinstance(st.session_state.comparison_results, dict):
             st.info(st.session_state.comparison_results['info'])

# --- TAB 2: Database View ---
with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("📚 Extracted Reference Data")
    if not st.session_state['extracted_data'].empty:
        df_display = st.session_state['extracted_data']
        df_view = df_display.copy()
        cols_to_map = ['Used before NEA', 'Used from DPEO', 'Used from CDPO']
        for col in cols_to_map:
            if col in df_view.columns:
                df_view[col] = df_view[col].astype(str).str.strip().str.upper() == 'Y'

        st.markdown(f"**Total Records:** `{len(df_display)}` | **Columns:** `{df_display.shape[1]}`")
        st.dataframe(
            df_view, 
            width="stretch", 
            height=600,
            column_config={
                "Used before NEA": st.column_config.CheckboxColumn("NEA", width="small"),
                "Used from DPEO": st.column_config.CheckboxColumn("DPEO", width="small"),
                "Used from CDPO": st.column_config.CheckboxColumn("CDPO", width="small"),
            }
        )
        st.caption(f"Local Cache: {FILE_PATH}")
    else:
        st.warning("No data extracted. Upload a file in the sidebar to view content.")
    st.markdown('</div>', unsafe_allow_html=True)
