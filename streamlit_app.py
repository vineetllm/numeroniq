import streamlit as st
import pandas as pd
import yfinance as yf
import re
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
import base64

import hashlib

# Utility function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Format: username: hashed_password
USER_CREDENTIALS = {
    "admin": hash_password("admin123"),
    "transleads": hash_password("leads27"),
    "vin": hash_password("vin69"),
}

# Check login status
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.title("🔐 Secure Access")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

# Require login before running the app
if not st.session_state.authenticated:
    login()
    st.stop()

# Set wide layout
st.set_page_config(page_title="Numeroniq", layout="wide")

# Inject CSS and JS to disable text selection and right-click
st.markdown("""
    <style>
    * {
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
        user-select: none !important;
    }

    /* Specifically target tables */
    div[data-testid="stTable"] {
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
        user-select: none !important;
    }

    /* Also target the scrollable dataframe area */
    .css-1wmy9hl, .css-1xarl3l {
        user-select: none !important;
    }
    </style>

    <script>
    document.addEventListener('contextmenu', event => event.preventDefault());
    </script>
    """, unsafe_allow_html=True)
# Disable right click with JavaScript
st.markdown("""
    <script>
    document.addEventListener('contextmenu', event => event.preventDefault());
    </script>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #e6cbb6, #fde6ef, #dcf7fc, #c2f0f7);
        }
        .block-container {
            background: radial-gradient(circle at top left, #e6cbb6, #fde6ef, #dcf7fc, #c2f0f7);
            padding: 2rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Define custom CSS for the table background
custom_css = """
<style>
.scroll-table {
    overflow-x: auto;
    max-height: 500px;
    border: 1px solid #ccc;
}

.scroll-table table {
    width: 100%;
    border-collapse: collapse;
    background-color: #f0f8ff; /* Light blue background */
}

.scroll-table th, .scroll-table td {
    padding: 8px;
    border: 1px solid #ddd;
    text-align: left;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load stock data
@st.cache_data
def load_stock_data():
    df = pd.read_excel("doc.xlsx")
    df['NSE LISTING DATE'] = pd.to_datetime(df['NSE LISTING DATE'], errors='coerce')
    df['BSE LISTING DATE'] = pd.to_datetime(df['BSE LISTING DATE'], errors='coerce')
    df['DATE OF INCORPORATION'] = pd.to_datetime(df['DATE OF INCORPORATION'], errors='coerce')
    return df

@st.cache_data(ttl=3600)
def load_excel_data(file):
    df = pd.read_excel(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

# Load numerology data
@st.cache_data
def load_numerology_data():
    df = pd.read_excel("numerology.xlsx")
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    return df

def calculate_destiny_number(date_obj):
    if pd.isnull(date_obj):
        return None, None
    digits = [int(ch) for ch in date_obj.strftime('%Y%m%d')]
    total = sum(digits)
    reduced = reduce_to_single_digit(total)
    return total, reduced

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, multi_level_index = False)
    return stock_data


def plot_candlestick_chart(stock_data, vertical_lines=None):

    import plotly.graph_objects as go
    import pandas as pd
    import streamlit as st

    # ✅ Normalize index for consistent date comparison
    stock_data.index = pd.to_datetime(stock_data.index).normalize()


    """
    Generate and return a candlestick chart using Plotly,
    with optional vertical lines on specific dates.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    stock_data.index = pd.to_datetime(stock_data.index).normalize()

    for date_str in vertical_lines:
        try:
            date_obj = pd.to_datetime(date_str).normalize()
            fig.add_vline(
                x=date_obj,
                line_width=2,
                line_dash="solid",
                line_color="black",
               
            )
        except Exception as e:
            print(f"Could not plot vertical line for {date_str}: {e}")


    
    fig.update_layout(
        title="Candlestick chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    return fig


# Load data
stock_df = load_stock_data()
numerology_df = load_numerology_data()


# === Chaldean Numerology Setup ===
chaldean_map = {
    1: 'A I J Q Y',
    2: 'B K R',
    3: 'C G L S',
    4: 'D M T',
    5: 'E H N X',
    6: 'U V W',
    7: 'O Z',
    8: 'F P'
}

# === Pythagorean Numerology Setup ===
pythagorean_map = {
    1: 'A J S',
    2: 'B K T',
    3: 'C L U',
    4: 'D M V',
    5: 'E N W',
    6: 'F O X',
    7: 'G P Y',
    8: 'H Q Z',
    9: 'I R'
}

char_to_num = {letter: num for num, letters in chaldean_map.items() for letter in letters.split()}

pythagorean_char_to_num = {
    letter: num for num, letters in pythagorean_map.items() for letter in letters.split()
}

def calculate_pythagorean_numerology(name):
    clean_name = re.sub(r'[^A-Za-z ]+', '', name)
    words = re.findall(r'\b[A-Za-z]+\b', clean_name)

    word_parts = []
    original_values = []

    for word in words:
        word_val = sum(pythagorean_char_to_num.get(char.upper(), 0) for char in word)
        reduced_val = reduce_to_single_digit(word_val)
        word_parts.append(f"{word_val}({reduced_val})")
        original_values.append(word_val)

    if not original_values:
        return None, None

    total_sum = sum(original_values)
    final_reduced = reduce_to_single_digit(total_sum)
    equation = f"{' + '.join(word_parts)} = {total_sum}({final_reduced})"
    return final_reduced, equation

def get_word_value(word):
    return sum(char_to_num.get(char.upper(), 0) for char in word)

def reduce_to_single_digit(n):
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n

def calculate_numerology(name):
    clean_name = re.sub(r'[^A-Za-z ]+', '', name)
    words = re.findall(r'\b[A-Za-z]+\b', clean_name)

    word_parts = []
    original_values = []

    for word in words:
        word_val = get_word_value(word)
        reduced_val = reduce_to_single_digit(word_val)
        word_parts.append(f"{word_val}({reduced_val})")
        original_values.append(word_val)

    if not original_values:
        return None, None

    total_sum = sum(original_values)
    final_reduced = reduce_to_single_digit(total_sum)

    equation = f"{' + '.join(word_parts)} = {total_sum}({final_reduced})"
    return final_reduced, equation

def calculate_chaldean_isin_numerology(isin):
    """
    Calculates Chaldean numerology for ISIN.
    Returns total and reduced value in format: 34(7)
    """
    if not isin:
        return None, None

    total = 0
    for char in isin:
        if char.isdigit():
            total += int(char)
        elif char.upper() in char_to_num:
            total += char_to_num[char.upper()]

    if total == 0:
        return None, None

    reduced = reduce_to_single_digit(total)
    return reduced, f"{total}({reduced})"

def calculate_pythagorean_isin_numerology(isin):
    """
    Calculates Pythagorean numerology for ISIN.
    Returns total and reduced value in format: 34(7)
    """
    if not isin:
        return None, None

    total = 0
    for char in isin:
        if char.isdigit():
            total += int(char)
        elif char.upper() in pythagorean_char_to_num:
            total += pythagorean_char_to_num[char.upper()]

    if total == 0:
        return None, None

    reduced = reduce_to_single_digit(total)
    return reduced, f"{total}({reduced})"


st.title("📊 Numeroniq")

st.html("""
<style>
[data-testid=stElementToolbarButton]:first-of-type {
    display: none;
}
</style>
""")

# === Toggle between filtering methods ===
st.sidebar.title("📊 Navigation")
filter_mode = st.sidebar.radio(
    "Choose Filter Mode:", 
    [
        "Company Overview", 
        "Numerology Date Filter", 
        "Filter by Sector/Symbol", 
        "Filter by Numerology",
        "Name Numerology", 
        "View Nifty/BankNifty OHLC", 
        "Equinox",
        "Moon",
        "Mercury",
        "Sun Number Dates",
        "Panchak"])

if filter_mode == "Filter by Sector/Symbol":
    # === Sector Filter ===
    sectors = stock_df['SECTOR'].dropna().unique()
    selected_sector = st.selectbox("Select Sector:", ["All"] + sorted(sectors))

    show_all_in_sector = st.checkbox("Show all companies in this sector", value=True)

    if selected_sector != "All":
        sector_filtered_df = stock_df[stock_df['SECTOR'] == selected_sector]
    else:
        sector_filtered_df = stock_df.copy()

    if not show_all_in_sector:
        filtered_symbols = sector_filtered_df['Symbol'].dropna().unique()
        selected_symbol = st.selectbox("Select Symbol:", sorted(filtered_symbols))
        company_data = sector_filtered_df[sector_filtered_df['Symbol'] == selected_symbol]
    else:
        company_data = sector_filtered_df

    # === Display Company Data ===
    if not company_data.empty:
        st.write("### Company Info")
        display_cols = company_data.drop(columns=['Series', 'Company Name', 'ISIN Code', 'IPO TIMING ON NSE'], errors='ignore')
        for col in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
            if col in display_cols.columns:
                display_cols[col] = display_cols[col].dt.strftime('%Y-%m-%d')
        # Convert DataFrame to HTML table
        html_table = display_cols.to_html(index=False, escape=False)

        # Embed HTML table in a scrollable container
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

        # Date choice: Single date or All Dates (NSE, BSE, Incorporation)
        date_choice = st.radio("Select Listing Date Source for Numerology:", 
                               ("NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION", "By All Dates"))

        if date_choice == "By All Dates":
            # If "By All Dates" is selected, we show three rows for each symbol
            combined_numerology = []
            for idx, row in company_data.iterrows():
                for date_column in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
                    date_val = row[date_column]
                    if pd.notnull(date_val):
                        numerology_row = numerology_df[numerology_df['date'] == pd.to_datetime(date_val)]
                        if not numerology_row.empty:
                            temp = numerology_row.copy()
                            temp['Symbol'] = row['Symbol']
                            temp['Date Type'] = date_column
                            temp['NSE age'] = row['NSE age']  # Add NSE AGE column
                            temp['BSE age'] = row['BSE age']  # Add BSE AGE column
                            temp['DOC age'] = row['DOC age']  # Add DOC AGE column
                            combined_numerology.append(temp)

            if combined_numerology:
                st.write(f"### Numerology Data for All Companies in {selected_sector} (Using All Dates)")
                all_numerology_df = pd.concat(combined_numerology, ignore_index=True)

                # Reorder columns: Symbol, Date Type, Date Used first
                cols = all_numerology_df.columns.tolist()
                cols = ['Symbol', 'Date Type', 'NSE age', 'BSE age', 'DOC age'] + [col for col in all_numerology_df.columns if col not in ['Symbol', 'Date Type', 'NSE age', 'BSE age', 'DOC age']]
                all_numerology_df = all_numerology_df[cols]

                # Convert DataFrame to HTML table
                html_table = all_numerology_df.to_html(index=False, escape=False)

                # Embed HTML table in a scrollable container
                st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

            else:
                st.warning("No numerology data found for selected dates across these companies.")
        
        else:
            # Handle the case where a single date type (NSE/BSE/Inc) is selected
            if len(company_data) == 1:
                # Single company selected — choose one date
                listing_date = pd.to_datetime(company_data[date_choice].values[0])
                if pd.notnull(listing_date):
                    st.write(f"### Numerology Data for {listing_date.strftime('%Y-%m-%d')}")
                    matched_numerology = numerology_df[numerology_df['date'] == listing_date].copy()
                    if not matched_numerology.empty:
                        matched_numerology['Symbol'] = company_data['Symbol'].values[0]
                        matched_numerology['Date Type'] = date_choice

                        if date_choice == "NSE LISTING DATE":
                            matched_numerology['NSE age'] = company_data['NSE age'].values[0]
                        elif date_choice == "BSE LISTING DATE":
                            matched_numerology['BSE age'] = company_data['BSE age'].values[0]
                        elif date_choice == "DATE OF INCORPORATION":
                            matched_numerology['DOC age'] = company_data['DOC age'].values[0]
                            
                        # Convert DataFrame to HTML table
                        html_table = matched_numerology.to_html(index=False, escape=False)

                        # Embed HTML table in a scrollable container
                        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No numerology data found for this date.")
                else:
                    st.warning(f"{date_choice} is not available for this company.")

            elif show_all_in_sector:
                # Multiple companies shown (whole sector) — apply one date field to all
                combined_numerology = []

                for idx, row in company_data.iterrows():
                    date_val = row[date_choice]
                    if pd.notnull(date_val):
                        numerology_row = numerology_df[numerology_df['date'] == pd.to_datetime(date_val)]
                        if not numerology_row.empty:
                            temp = numerology_row.copy()
                            temp['Symbol'] = row['Symbol']
                            temp['Date Type'] = date_choice

                            if date_choice == "NSE LISTING DATE":
                                temp['NSE age'] = row['NSE age']
                            elif date_choice == "BSE LISTING DATE":
                                temp['BSE age'] = row['BSE age']
                            elif date_choice == "DATE OF INCORPORATION":
                                temp['DOC age'] = row['DOC age']

                            combined_numerology.append(temp)

                if combined_numerology:
                    st.write(f"### Numerology Data for All Companies in {selected_sector} (Using {date_choice})")
                    all_numerology_df = pd.concat(combined_numerology, ignore_index=True)

                    # Move Symbol, Date Type and relevant Age column to front
                    cols_to_front = ['Symbol', 'Date Type']
                    if date_choice == "NSE LISTING DATE":
                        cols_to_front.append('NSE age')
                    elif date_choice == "BSE LISTING DATE":
                        cols_to_front.append('BSE age')
                    elif date_choice == "DATE OF INCORPORATION":
                        cols_to_front.append('DOC age')

                    all_cols = cols_to_front + [col for col in all_numerology_df.columns if col not in cols_to_front]
                    
                    # Convert DataFrame to HTML table
                    html_table = all_numerology_df[all_cols].to_html(index=False, escape=False)

                    # Embed HTML table in a scrollable container
                    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

                else:
                    st.warning("No numerology data found for selected date field across these companies.")

            else:
                # Multiple companies manually selected but show_all_in_sector is False
                st.info("Select a single symbol (uncheck the box) to see numerology data.")

    else:
        st.warning("No matching data found.")

elif filter_mode == "Numerology Date Filter":
    st.subheader("📅 Filter Numerology Data by Date")

    # Parse and clean the date column
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], dayfirst=True, errors='coerce')
    numerology_df = numerology_df.dropna(subset=['date'])

    # Get year range
    min_year = numerology_df['date'].dt.year.min()
    max_year = numerology_df['date'].dt.year.max()

    # Build decade periods
    periods = []
    for start in range((min_year // 10) * 10, (max_year // 10 + 1) * 10, 10):
        end = start + 9
        periods.append(f"{start}-{end}")

    # Step 1: Select Time Period
    selected_period = st.selectbox("Select Time Period", periods)
    start_year, end_year = map(int, selected_period.split('-'))
    period_start = pd.to_datetime(f"{start_year}-01-01")
    period_end = pd.to_datetime(f"{end_year}-12-31")

    # Step 2: Let user refine with date pickers within the selected period
    date_range = numerology_df[(numerology_df['date'] >= period_start) & (numerology_df['date'] <= period_end)]
    if date_range.empty:
        st.warning("No data available in this period.")
        st.stop()

    min_date = date_range['date'].min().date()
    max_date = date_range['date'].max().date()

    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)

    # Step 3: Filter data based on custom date range within the period
    filtered = date_range[
        (date_range['date'] >= pd.to_datetime(start_date)) &
        (date_range['date'] <= pd.to_datetime(end_date))
    ]

    # Step 4: Additional filters
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        bn_filter = st.selectbox("BN", options=['All'] + numerology_df['BN'].dropna().unique().tolist(), index=0)
    with col2:
        dn_filter = st.selectbox("DN (Formatted)", options=['All'] + numerology_df['DN (Formatted)'].dropna().unique().tolist(), index=0)
    with col3:
        sn_filter = st.selectbox("SN", options=['All'] + numerology_df['SN'].dropna().unique().tolist(), index=0)
    with col4:
        hp_filter = st.selectbox("HP", options=['All'] + numerology_df['HP'].dropna().unique().tolist(), index=0)
    with col5:
        day_number_filter = st.selectbox("Day Number", options=['All'] + numerology_df['Day Number'].dropna().unique().tolist(), index=0)

    if bn_filter != 'All':
        filtered = filtered[filtered['BN'] == bn_filter]
    if dn_filter != 'All':
        filtered = filtered[filtered['DN (Formatted)'] == dn_filter]
    if sn_filter != 'All':
        filtered = filtered[filtered['SN'] == sn_filter]
    if hp_filter != 'All':
        filtered = filtered[filtered['HP'] == hp_filter]
    if day_number_filter != 'All':
        filtered = filtered[filtered['Day Number'] == day_number_filter]

    # Display filtered data
    st.write(f"Showing {len(filtered)} records from **{start_date}** to **{end_date}**")

    html_table = filtered.to_html(index=False, escape=False)
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Filter by Numerology":
    st.markdown("### 🔢 Filter by Numerology Values (Live & Horizontal Layout)")

    # Step 1: Ask how to match date: NSE/BSE/Inc
    date_match_option = st.selectbox("Select Date Type to Match Companies:", 
                                 ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"])

    # Step 2: Start with full numerology data
    filtered_numerology = numerology_df.copy()
    # Calculate DN dynamically
    dn_values = filtered_numerology['date'].apply(calculate_destiny_number)
    filtered_numerology['DN Raw'] = dn_values.apply(lambda x: x[0])
    filtered_numerology['DN'] = dn_values.apply(lambda x: x[1])
    filtered_numerology['DN (Formatted)'] = filtered_numerology.apply(lambda row: f"({row['DN Raw']}){row['DN']}" if pd.notnull(row['DN Raw']) else None, axis=1)


    # Prepare layout
    col1, col2, col3, col4, col5 = st.columns(5)

    # === BN Filter ===
    with col1:
        bn_options = ["All"] + sorted(numerology_df['BN'].dropna().unique())
        selected_bn = st.selectbox("BN", bn_options)
        if selected_bn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['BN'] == selected_bn]

    # === DN Filter ===
    with col2:
        dn_options = ["All"] + sorted(filtered_numerology['DN'].dropna().unique())
        selected_dn = st.selectbox("DN", dn_options)
        if selected_dn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['DN'] == selected_dn]

    # === SN Filter ===
    with col3:
        sn_options = ["All"] + sorted(filtered_numerology['SN'].dropna().unique())
        selected_sn = st.selectbox("SN", sn_options)
        if selected_sn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['SN'] == selected_sn]

    # === HP Filter ===
    with col4:
        hp_options = ["All"] + sorted(filtered_numerology['HP'].dropna().unique())
        selected_hp = st.selectbox("HP", hp_options)
        if selected_hp != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['HP'] == selected_hp]

    # === Day Number Filter ===
    with col5:
        dayn_options = ["All"] + sorted(filtered_numerology['Day Number'].dropna().unique())
        selected_dayn = st.selectbox("Day Number", dayn_options)
        if selected_dayn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['Day Number'] == selected_dayn]


    # Create a mapping of dates to numerology rows (after filter)
    filtered_numerology_map = filtered_numerology.set_index('date')

    # Loop through stock_df and match per company by selected date
    matching_records = []

    for _, row in stock_df.iterrows():
        match_date = row.get(date_match_option)
        if pd.notnull(match_date) and match_date in filtered_numerology_map.index:
            numerology_match = filtered_numerology_map.loc[match_date]
        
            # Handle multiple matches (if any) from numerology_df
            if isinstance(numerology_match, pd.DataFrame):
                numerology_match = numerology_match.iloc[0]

            combined_row = row.to_dict()
            combined_row.update(numerology_match.to_dict())
            combined_row['Matching Date Source'] = date_match_option
            matching_records.append(combined_row)

    # Create final DataFrame
    matching_stocks = pd.DataFrame(matching_records)

    st.markdown("### 🎯 Matching Companies")

    if not matching_stocks.empty:

        display_cols = matching_stocks.drop(columns=['Series', 'Company Name', 'ISIN Code', 'IPO TIMING ON NSE'], errors='ignore')

        # Format for display
        for col in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
            if col in display_cols.columns:
                display_cols[col] = display_cols[col].dt.strftime('%Y-%m-%d')

        # Optional: format date columns
        for col in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
            if col in matching_stocks.columns:
                matching_stocks[col] = pd.to_datetime(matching_stocks[col], errors='coerce').dt.strftime('%Y-%m-%d')

        # Reorder to show date source
        cols_order = ['Symbol', date_match_option, 'BN', 'DN (Formatted)', 'SN', 'HP', 'Day Number'] + \
            [col for col in matching_stocks.columns if col not in ['Symbol', 'Matching Date Source', date_match_option, 'BN', 'DN', 'DN (Formatted)', 'SN', 'HP', 'Day Number']]


        # Convert DataFrame to HTML table
        html_table = matching_stocks.to_html(index=False, escape=False)

        # Embed HTML table in a scrollable container
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    else:
        st.info("No companies found with matching numerology dates.")

elif filter_mode == "Name Numerology":
    st.subheader("🔢 Name Numerology")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        use_ltd = st.radio(
            "Include 'Ltd' or 'Limited'?",
            ["Yes", "No"],
            index=1
        )

    with col2:
        use_is_prefix = st.radio(
            "Include 'IN' prefix in ISIN numerology?",
            ["Yes", "No"],
            index=0
        )

    with col3:
        numerology_system = st.radio(
            "Numerology System:",
           ["Chaldean", "Pythagoras", "Both"]
        )


    numerology_data = []

    for _, row in stock_df.iterrows():
        company_original = row['Company Name']
        symbol = str(row['Symbol'])
        isin_code = str(row['ISIN Code']) 

        # Remove 'Ltd' or 'Limited' if user chose "No"
        if use_ltd == "No":
            company_clean = re.sub(r'\b(Ltd|Limited)\b', '', company_original, flags=re.IGNORECASE).strip()
        else:
            company_clean = company_original

        entry = {
            'Symbol': row['Symbol'],
            'Company Name': company_original,
            'ISIN Code': isin_code,  # Add ISIN code for display
        }

        if numerology_system in ["Chaldean", "Both"]:
            ch_company_num, ch_company_eq = calculate_numerology(company_clean)
            ch_symbol_num, ch_symbol_eq = calculate_numerology(symbol)
            isin_to_use = isin_code if use_is_prefix == "Yes" else isin_code[2:]
            ch_isin_num, ch_isin_eq = calculate_chaldean_isin_numerology(isin_to_use)


            entry['Chaldean Eqn (Company Name)'] = ch_company_eq
            entry['Chaldean Eqn (Symbol)'] = ch_symbol_eq
            entry['Chaldean Eqn (ISIN Code)'] = ch_isin_eq

        if numerology_system in ["Pythagoras", "Both"]:
            py_company_num, py_company_eq = calculate_pythagorean_numerology(company_clean)
            py_symbol_num, py_symbol_eq = calculate_pythagorean_numerology(symbol)
            isin_to_use = isin_code if use_is_prefix == "Yes" else isin_code[2:]
            py_isin_num, py_isin_eq = calculate_pythagorean_isin_numerology(isin_to_use) 
            entry['Pythagoras Eqn (Company Name)'] = py_company_eq
            entry['Pythagoras Eqn (Symbol)'] = py_symbol_eq
            entry['Pythagoras Eqn (ISIN Code)'] = py_isin_eq 

        numerology_data.append(entry)


    numerology_df_display = pd.DataFrame(numerology_data)


    # === Filters ===
    col1, col2 = st.columns(2)

    with col1:
        company_filter = st.selectbox(
            "Select Company (or choose All)",
            options=["All"] + sorted(numerology_df_display['Company Name'].unique())
        )


    filtered_df = numerology_df_display.copy()

    if company_filter != "All":
        filtered_df = filtered_df[filtered_df['Company Name'] == company_filter]

    if numerology_system in ["Chaldean", "Both"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            ch_company_totals = numerology_df_display['Chaldean Eqn (Company Name)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_ch_company = st.selectbox("Chaldean Total (Company Name)", ["All"] + sorted(ch_company_totals.dropna().unique()))

        with col2:
            ch_symbol_totals = numerology_df_display['Chaldean Eqn (Symbol)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_ch_symbol = st.selectbox("Chaldean Total (Symbol)", ["All"] + sorted(ch_symbol_totals.dropna().unique()))

        with col3:
            ch_isin_totals = numerology_df_display['Chaldean Eqn (ISIN Code)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_ch_isin = st.selectbox("Chaldean Total (ISIN Code)", ["All"] + sorted(ch_isin_totals.dropna().unique()))

        if selected_ch_company != "All":
            filtered_df = filtered_df[
                filtered_df['Chaldean Eqn (Company Name)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_ch_company)
            ]

        if selected_ch_symbol != "All":
            filtered_df = filtered_df[
                filtered_df['Chaldean Eqn (Symbol)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_ch_symbol)
            ]

        if selected_ch_isin != "All":
            filtered_df = filtered_df[
                filtered_df['Chaldean Eqn (ISIN Code)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_ch_isin)
            ]

    if numerology_system in ["Pythagoras", "Both"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            py_company_totals = numerology_df_display['Pythagoras Eqn (Company Name)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_py_company = st.selectbox("Pythagoras Total (Company Name)", ["All"] + sorted(py_company_totals.dropna().unique()))

        with col2:
            py_symbol_totals = numerology_df_display['Pythagoras Eqn (Symbol)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_py_symbol = st.selectbox("Pythagoras Total (Symbol)", ["All"] + sorted(py_symbol_totals.dropna().unique()))

        with col2:
            py_isin_totals = numerology_df_display['Pythagoras Eqn (ISIN Code)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_py_isin = st.selectbox("Pythagoras Total (ISIN Code)", ["All"] + sorted(py_isin_totals.dropna().unique()))


        if selected_py_company != "All":
            filtered_df = filtered_df[
                filtered_df['Pythagoras Eqn (Company Name)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_py_company)
            ]

        if selected_py_symbol != "All":
            filtered_df = filtered_df[
                filtered_df['Pythagoras Eqn (Symbol)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_py_symbol)
            ]

        if selected_py_isin != "All":
            filtered_df = filtered_df[
                filtered_df['Pythagoras Eqn (ISIN Code)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_py_isin)
            ]


    
    # Convert DataFrame to HTML table
    html_table = filtered_df.to_html(index=False, escape=False)

    # Embed HTML table in a scrollable container
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Company Overview":
    st.title("🏠 Company Overview")

    # Prepare searchable list for suggestions
    search_options = stock_df['Symbol'].dropna().tolist() + stock_df['Company Name'].dropna().tolist()
    search_options = sorted(set(search_options))

    user_input = st.selectbox("Search by Symbol or Company Name:", options=[""] + search_options)

    if user_input:
        # Case-insensitive match
        company_info = stock_df[
            (stock_df['Symbol'].str.lower() == user_input.lower()) |
            (stock_df['Company Name'].str.lower() == user_input.lower())
        ]

        if not company_info.empty:
            row = company_info.iloc[0]

            # --- Line 1: Sector, Sub-sector, ISIN Code ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Sector:** {row.get('SECTOR', 'N/A')}")
            with col2:
                st.markdown(f"**Sub-Sector:** {row.get('SUB SECTOR', 'N/A')}")
            with col3:
                st.markdown(f"**ISIN Code:** {row.get('ISIN Code', 'N/A')}")

            # --- Line 2: Dates ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Date of Incorporation:** {row['DATE OF INCORPORATION'].date() if pd.notnull(row['DATE OF INCORPORATION']) else 'N/A'}")
            with col2:
                st.markdown(f"**NSE Listing Date:** {row['NSE LISTING DATE'].date() if pd.notnull(row['NSE LISTING DATE']) else 'N/A'}")
            with col3:
                st.markdown(f"**BSE Listing Date:** {row['BSE LISTING DATE'].date() if pd.notnull(row['BSE LISTING DATE']) else 'N/A'}")

            # --- Line 3: Ages ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**DOC Age:** {row.get('DOC age', 'N/A')}")
            with col2:
                st.markdown(f"**NSE Age:** {row.get('NSE age', 'N/A')}")
            with col3:
                st.markdown(f"**BSE Age:** {row.get('BSE age', 'N/A')}")

            # --- Line 4: Name Numerology ---
            st.markdown("### 🔢 Name Numerology")

            use_ltd_home = st.radio(
                "Include 'Ltd' or 'Limited' in company name (if present)?",
                ["Yes", "No"],
                index=1,
                key="home_ltd"
            )

            use_in_prefix_home = st.radio(
                "Include 'IN' prefix in ISIN code (if present)?",
                ["Yes", "No"],
                index=1,
                key="home_isin"
            )

            isin_code = str(row.get("ISIN Code", ""))

            company_name_original = str(row['Company Name'])
            symbol_name = str(row['Symbol'])
            isin_code = str(row['ISIN Code'])

            if use_ltd_home == "No":
                company_clean = re.sub(r'\b(Ltd|Limited)\b', '', company_name_original, flags=re.IGNORECASE).strip()
            else:
                company_clean = company_name_original

            if use_in_prefix_home == "Yes":
                isin_to_use = isin_code
            else:
                isin_to_use = isin_code[2:] if isin_code.startswith("IN") else isin_code

            # Chaldean system
            ch_company_num, ch_company_eq = calculate_numerology(company_clean)
            ch_symbol_num, ch_symbol_eq = calculate_numerology(symbol_name)
            ch_isin_num, ch_isin_eq = calculate_chaldean_isin_numerology(isin_to_use)

            # Pythagorean system
            py_company_num, py_company_eq = calculate_pythagorean_numerology(company_clean)
            py_symbol_num, py_symbol_eq = calculate_pythagorean_numerology(symbol_name)
            py_isin_num, py_isin_eq = calculate_pythagorean_isin_numerology(isin_to_use)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Chaldean Eqn (Company Name):** {ch_company_eq}")
                st.markdown(f"**Chaldean Eqn (Symbol):** {ch_symbol_eq}")
                st.markdown(f"**Chaldean Eqn (ISIN Code):** {ch_isin_eq}")

            with col2:
                st.markdown(f"**Pythagoras Eqn (Company Name):** {py_company_eq}")
                st.markdown(f"**Pythagoras Eqn (Symbol):** {py_symbol_eq}")
                st.markdown(f"**Pythagoras Eqn (ISIN Code):** {py_isin_eq}")

            # --- Line 5: Zodiac Signs ---
            st.markdown("### ♈ Zodiac Information (Based on Dates)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**DOC Zodiac Sign:** {row.get('DOC zodiac sign', 'N/A')}")
                doc_zodiac_number = row.get('DOC zodiac number', 'N/A')
                if isinstance(doc_zodiac_number, float) and doc_zodiac_number.is_integer():
                    doc_zodiac_number = int(doc_zodiac_number)
                st.markdown(f"**DOC Zodiac Number:** {doc_zodiac_number}")


            with col2:
                st.markdown(f"**NSE Zodiac Sign:** {row.get('NSE zodiac sign', 'N/A')}")
                st.markdown(f"**NSE Zodiac Number:** {row.get('NSE zodiac number', 'N/A')}")

            with col3:
                st.markdown(f"**BSE Zodiac Sign:** {row.get('BSE zodiac sign', 'N/A')}")
                st.markdown(f"**BSE Zodiac Number:** {row.get('BSE zodiac number', 'N/A')}")

            # --- Numerology Selection for Home Page ---
            st.markdown("### 🔢 Numerology Data Based on Selected Date")

            # Step 1: Ask user for date type preference
            date_match_option = st.selectbox(
                "Select Date Type to View Numerology Data:",
                ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION", "All Dates"]
            )

            selected_row = row  # Already fetched from earlier using user_input

            date_types = (
                ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"]
                if date_match_option == "All Dates"
                else [date_match_option]
            )

            vertical_lines = []

            for dt_type in date_types:
                match_date = selected_row.get(dt_type)
                st.markdown(f"#### 📅 Numerology for {dt_type}: {match_date.date() if pd.notnull(match_date) else 'N/A'}")

                if pd.notnull(match_date):
                    match_date = pd.to_datetime(match_date)
                    numerology_row = numerology_df[numerology_df['date'] == match_date]
                    if not numerology_row.empty:
                        row_data = numerology_row.iloc[0]

                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.markdown(f"**BN:** {row_data.get('BN', 'N/A')}")
                        with col2:
                            st.markdown(f"**DN (Formatted):** {row_data.get('DN (Formatted)', 'N/A')}")
                        with col3:
                            st.markdown(f"**SN:** {row_data.get('SN', 'N/A')}")
                        with col4:
                            st.markdown(f"**HP:** {row_data.get('HP', 'N/A')}")
                        with col5:
                            st.markdown(f"**Day Number:** {row_data.get('Day Number', 'N/A')}")

                        # SN-based vertical line mapping
                        sn_vertical_lines = {
                            1: ["2025-05-05", "2025-05-07", "2025-05-08", "2025-05-10"],
                            2: ["2025-05-03", "2025-05-08", "2025-05-09", "2025-05-13"],
                            3: ["2025-05-06", "2025-05-10", "2025-05-11"],
                            4: ["2025-05-01", "2025-05-04", "2025-05-11", "2025-05-12"],
                            5: ["2025-05-02", "2025-05-05", "2025-05-08", "2025-05-12"],
                            6: ["2025-05-01", "2025-05-03", "2025-05-11", "2025-05-13"],
                            7: ["2025-05-04", "2025-05-14"],
                            8: ["2025-05-05", "2025-05-07", "2025-05-09"],
                            9: ["2025-05-02", "2025-05-06", "2025-05-11"]
                        }

                        # Extract SN value from numerology row
                        sn_value = row_data.get('SN', None)
                        if sn_value in sn_vertical_lines:
                            vertical_lines.extend(sn_vertical_lines[sn_value])


                else:
                    st.info(f"No numerology data available for {dt_type}.")
            else:
                st.info(f"No date available for {dt_type}.")

            vertical_lines = [pd.to_datetime(d) for d in vertical_lines]

            # --- Candlestick Chart (After Zodiac Info) ---
            st.markdown("### 📈 Stock Price Candlestick Chart")

            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("today").normalize())
            ticker = str(row['Symbol']).upper() + ".NS"  # Get the company's symbol

            if start_date and end_date:
                stock_data = get_stock_data(ticker, start_date, end_date)
                if not stock_data.empty:
                    chart = plot_candlestick_chart(stock_data, vertical_lines=vertical_lines)
                    st.plotly_chart(chart)
                else:
                    st.warning("No data available for the selected date range.")
        else:
            st.warning("No matching company found.")

elif filter_mode == "View Nifty/BankNifty OHLC":
    st.subheader("📈 Nifty & BankNifty OHLC Viewer")

    import yfinance as yf
    from datetime import datetime

    index_choice = st.selectbox("Select Index:", ["Nifty 50", "Bank Nifty"])

    if index_choice == "Nifty 50":
        file = "nifty.xlsx"
        symbol = "^NSEI"
    else:
        file = "banknifty.xlsx"
        symbol = "^NSEBANK"

    @st.cache_data(ttl=3600)
    def load_excel_data(file):
        df = pd.read_excel(file, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    excel_data = load_excel_data(file)
    last_excel_date = excel_data.index[-1].date()

    # Fetch data from yfinance only after last_excel_date
    @st.cache_data(ttl=3600)
    def fetch_yfinance_data(symbol, start_date):
        yf_data = yf.download(symbol, start=start_date, interval="1d", multi_level_index= False)[['Open', 'High', 'Low', 'Close', 'Volume']]
        return yf_data

    today = datetime.today().date()
    if last_excel_date < today:
        yf_data = fetch_yfinance_data(symbol, start_date=last_excel_date + pd.Timedelta(days=1))
        full_data = pd.concat([excel_data, yf_data])
    else:
        full_data = excel_data.copy()

    # Calculate Volatility % and Close %
    full_data['Volatility %'] = ((full_data['High'] - full_data['Low']) / full_data['Low']) * 100
    full_data['Close %'] = full_data['Close'].pct_change() * 100

    full_data['Volatility %'] = full_data['Volatility %'].round(2)
    full_data['Close %'] = full_data['Close %'].round(2)

    full_data['Day'] = full_data.index.day
    full_data['Month'] = full_data.index.month


    # Reorder columns: Volatility % and Close % first
    reordered_cols = ['Volatility %', 'Close %', 'Open', 'High', 'Low', 'Close', 'Vol(in M)']
    full_data = full_data[reordered_cols]

    # Default date range: last 90 days
    default_end_date = full_data.index.max()
    default_start_date = default_end_date - timedelta(days=90)

    # Let user choose range
    start_date = st.date_input("Start Date", value=default_start_date,
                               min_value=full_data.index.min().date(), 
                               max_value=full_data.index.max().date())
    end_date = st.date_input("End Date", value=default_end_date,
                             min_value=full_data.index.min().date(),
                             max_value=full_data.index.max().date())

    # Ensure end_date is not earlier than start_date
    if start_date > end_date:
        st.error("End Date must be after Start Date")
    else:
        # Convert to datetime and filter
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        full_data = full_data.loc[(full_data.index >= start_dt) & (full_data.index <= end_dt)]
        full_data = full_data.sort_index(ascending=False)



    # Filters in horizontal layout
    st.markdown("### 🔍 Filter OHLC Table")
    col1, col2, col3 = st.columns(3)

    with col1:
        vol_op = st.selectbox("Volatility Operator", ["All", "<", "<=", ">", ">=", "=="])
        vol_val = st.number_input("Volatility % Value", value=2.0, step=0.1)

    with col2:
        close_op = st.selectbox("Close % Operator", ["All", "<", "<=", ">", ">=", "=="])
        close_val = st.number_input("Close % Value", value=0.5, step=0.1)

    # Checkbox to enable/disable Day & Month filter
    apply_day_month_filter = st.checkbox("🗓️ Filter by Day and Month", value=False)


    filtered_data = full_data.copy()

    if vol_op != "All":
        filtered_data = filtered_data.query(f"Volatility % {vol_op} @vol_val")

    if close_op != "All":
        filtered_data = filtered_data.query(f"Close % {close_op} @close_val")

    # Optional Day & Month filter
    if apply_day_month_filter:
        st.markdown("### 📅 Filter by Day and Month")
        dcol1, dcol2 = st.columns(2)

        with dcol1:
            filter_day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=1)

        with dcol2:
            filter_month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                format_func=lambda x: datetime(1900, x, 1).strftime('%B')
            )

        filtered_data = filtered_data[
            (filtered_data.index.day == filter_day) &
            (filtered_data.index.month == filter_month)
        ]


    # Merge numerology data with OHLC data on date
    numerology_aligned = numerology_df.copy()
    numerology_aligned = numerology_aligned.set_index('date')
    numerology_aligned.index = pd.to_datetime(numerology_aligned.index)
    
    full_data_merged = filtered_data.merge(numerology_aligned, left_index=True, right_index=True, how='left')



    # Numerology filters
    st.markdown("### 🧮 Numerology Filters")
    ncol1, ncol2, ncol3, ncol4, ncol5 = st.columns(5)

    with ncol1:
        bn_filter = st.selectbox("BN", ["All"] + sorted(numerology_df['BN'].dropna().unique()))

    with ncol2:
        dn_filter = st.selectbox("DN (Formatted)", ["All"] + sorted(numerology_df['DN (Formatted)'].dropna().unique()))

    with ncol3:
        sn_filter = st.selectbox("SN", ["All"] + sorted(numerology_df['SN'].dropna().unique()))

    with ncol4:
        hp_filter = st.selectbox("HP", ["All"] + sorted(numerology_df['HP'].dropna().unique()))

    with ncol5:
        dayn_filter = st.selectbox("Day Number", ["All"] + sorted(numerology_df['Day Number'].dropna().unique()))

    # Apply numerology filters
    filtered_merged = full_data_merged.copy()
    if bn_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['BN'] == bn_filter]
    if dn_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['DN (Formatted)'] == dn_filter]
    if sn_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['SN'] == sn_filter]
    if hp_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['HP'] == hp_filter]
    if dayn_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['Day Number'] == dayn_filter]

    # Display filtered, aligned data
    st.markdown("### 🔢 OHLC + Numerology Alignment")
    
    # Reorder columns
    ordered_cols = ['Volatility %', 'Close %', 'Open', 'High', 'Low', 'Close', 'Vol(in M)']
    numerology_cols = ['BN', 'DN (Formatted)', 'SN', 'HP', 'Day Number', 'BN Planet','DN Planet', 'SN Planet', 'HP Planet', 'Day Number Planet']
    # Include date as a column if it's not already (currently index)
    filtered_merged_reset = filtered_merged.reset_index()
    

    # Final column order
    final_order = ['Date'] + numerology_cols + ordered_cols
    existing_cols = [col for col in final_order if col in filtered_merged_reset.columns]

    # Desired column order (adjust as needed if columns vary)
    desired_order = [
        'Date',
        'BN', 'DN (Formatted)', 
        'SN', 'HP', 
        'Day Number', 'BN Planet',
        'DN Planet',  'SN Planet',
        'HP Planet', 'Day Number Planet',
        'Volatility %', 'Close %',
        'Open', 'High',
        'Low', 'Close', 'Vol(in M)'
    ]
    
    # === Color-Coded Rows Based on Date ===

    # Define target dates as (month, day)
    primary_dates = {(3, 20),(3, 21), (6, 20), (6, 21), (9, 22), (9, 23), (12, 21), (12, 22)}
    secondary_dates = {(2, 4), (5, 6), (8, 8), (11, 7)}

    def highlight_rows(row):
        date = pd.to_datetime(row['Date'])
        month_day = (date.month, date.day)
        if month_day in primary_dates:
            return 'background-color: lightgreen'
        elif month_day in secondary_dates:
            return 'background-color: lightsalmon'
        else:
            return ''

    # Apply styles row-wise and format numbers
    styled_df = (
        filtered_merged_reset[existing_cols]
        .style
        .apply(lambda row: [highlight_rows(row)] * len(row), axis=1)
        .format(precision=2)  # format all numeric columns to 2 decimal places
    )


    # Render as HTML
    html_table = styled_df.to_html()

    # Display
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    

    if st.checkbox("📊 Show Candlestick Chart"):
        if not filtered_data.empty:
            candlestick = go.Figure(data=[go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])

            candlestick.update_layout(
                title='Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                height=600
            )

            st.plotly_chart(candlestick, use_container_width=True)
        else:
            st.warning("No data available for selected filters to display candlestick chart.")
    
elif filter_mode == "Equinox":
    st.subheader("📊 Nifty/BankNifty Report for Primary & Secondary Dates")

    # Step 1: Choose Index
    index_choice = st.selectbox("Choose Index", ["Nifty 50", "Bank Nifty"], key="econ_index")

    file = "nifty.xlsx" if index_choice == "Nifty 50" else "banknifty.xlsx"

    @st.cache_data(ttl=3600)
    def load_excel_data_for_report(file):
        df = pd.read_excel(file, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    ohlc_data = load_excel_data_for_report(file)

    # Recalculate Volatility % and Close %
    ohlc_data['Volatility %'] = ((ohlc_data['High'] - ohlc_data['Low']) / ohlc_data['Low']) * 100
    ohlc_data['Close %'] = ohlc_data['Close'].pct_change() * 100

    # Round for display
    ohlc_data['Volatility %'] = ohlc_data['Volatility %'].round(2)
    ohlc_data['Close %'] = ohlc_data['Close %'].round(2)

    # Ensure numerology dates are datetime
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce')
    numerology_data = numerology_df.set_index('date')

    # Step 2: Define all valid numerology dates
    all_dates = numerology_df['date'].dropna().dt.date.unique()
    all_dates = sorted(pd.to_datetime(all_dates))

    # Step 3: Time Period Selection
    st.markdown("### 🗓️ Select Time Period")
    min_date = min(all_dates)
    max_date = max(all_dates)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
    else:
        # Define dates of interest
        primary_dates = {(3, 21), (6, 20), (6, 21), (9, 22), (9, 23), (12, 21), (12, 22)}
        secondary_dates = {(2, 4), (5, 6), (8, 8), (11, 7)}

        def classify_date(dt):
            m, d = dt.month, dt.day
            if (m, d) in primary_dates:
                return "Primary"
            elif (m, d) in secondary_dates:
                return "Secondary"
            return None

        # Filter numerology dates by selected period
        filtered_dates = [dt for dt in all_dates if start_date <= dt.date() <= end_date]

        report_rows = []
        for date in filtered_dates:
            tag = classify_date(date)
            if tag:
                row = {"Date": date, "Category": tag}

                # OHLC if available
                if date in ohlc_data.index:
                    row.update(ohlc_data.loc[date].to_dict())
                else:
                    for col in ['Open', 'High', 'Low', 'Close', 'Vol(in M)', 'Volatility %', 'Close %']:
                        row[col] = float('nan')

                # Add numerology info if available
                if date in numerology_data.index:
                    row.update(numerology_data.loc[date].to_dict())

                report_rows.append(row)

        if report_rows:
            final_df = pd.DataFrame(report_rows)
            final_df = final_df.sort_values("Date", ascending=False).reset_index(drop=True)
            final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')

            # Define float columns to round
            float_cols = ['Open', 'High', 'Low', 'Close', 'Vol(in M)', 'Volatility %', 'Close %']
            rounded_df = final_df.copy()

            for col in float_cols:
                if col in rounded_df.columns:
                    rounded_df[col] = rounded_df[col].astype(float).round(2)

            # Create formatter for 2 decimal places
            formatter_dict = {col: "{:.2f}" for col in float_cols if col in rounded_df.columns}

            # === Apply Row Highlights Based on Primary/Secondary ===
            primary_dates = {(3, 21), (6, 20), (6, 21), (9, 22), (9, 23), (12, 21), (12, 22)}
            secondary_dates = {(2, 4), (5, 6), (8, 8), (11, 7)}

            def highlight_econ_rows(row):
                date = pd.to_datetime(row['Date'], errors='coerce')
                if pd.isna(date): return ''
                md = (date.month, date.day)
                if md in primary_dates:
                    return 'background-color: #d1fab8'
                elif md in secondary_dates:
                    return 'background-color: #ffa868'
                else:
                    return ''

            # Apply styling + formatting
            styled_df = rounded_df.style \
                .apply(lambda row: [highlight_econ_rows(row)] * len(row), axis=1) \
                .format(formatter_dict)

            # Render to HTML
            html_table = styled_df.to_html()

            # Display
            st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

        else:
            st.info("No primary or secondary dates found in selected range.")

elif filter_mode == "Moon":
    st.header("🌑 Moon Phase Analysis")

    # Load moon data
    moon_df = pd.read_excel("moon.xlsx")
    moon_df['Date'] = pd.to_datetime(moon_df['Date'], dayfirst=True)
    moon_df = moon_df.sort_values('Date')

    # Load stock symbols from doc.xlsx
    doc_df = pd.read_excel("doc.xlsx")
    available_symbols = sorted(doc_df['Symbol'].dropna().unique().tolist())

    # Load numerology
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], dayfirst=True)

    # Moon Phase & Date
    phase_choice = st.selectbox("Select Moon Phase:", ["Amavasya", "Poornima"])
    phase_filtered = moon_df[moon_df['A/P'].str.lower() == phase_choice.lower()]
    available_dates = phase_filtered['Date'].dt.strftime("%Y-%m-%d").tolist()
    selected_date_str = st.selectbox(f"Select a {phase_choice} Date:", available_dates)
    selected_date = pd.to_datetime(selected_date_str)

    # Moon Info
    match = moon_df[moon_df['Date'].dt.date == selected_date.date()]
    if match.empty:
        st.error("Selected date not found in moon data.")
        st.stop()

    selected_row = match.iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Degree:** {selected_row['Degree']}")
    with col2:
        st.markdown(f"**Time:** {selected_row['Time']}")
    with col3:
        st.markdown(f"**Paksh:** {selected_row['Paksh']}")

    # Find next moon date
    future_dates = moon_df[moon_df['Date'] > selected_date]
    next_date = future_dates.iloc[0]['Date'] if not future_dates.empty else selected_date + pd.Timedelta(days=15)
    st.markdown(f"### 📅 Period: {selected_date.date()} to {next_date.date()}")

    st.subheader("📈 Symbol OHLC + Numerology")

    # --- SYMBOL SECTION ---
    selected_symbol = st.selectbox("Select Stock Symbol:", available_symbols)

    listing_row = doc_df[doc_df['Symbol'] == selected_symbol]
    if listing_row.empty or pd.isnull(listing_row.iloc[0]['DATE OF INCORPORATION']):
        st.warning("Listing date unavailable.")
    else:
        listing_date = pd.to_datetime(listing_row.iloc[0]['DATE OF INCORPORATION'])

        if selected_date < listing_date:
            st.warning(f"{selected_symbol} was not listed on {selected_date.date()}")
        else:
            ticker = selected_symbol + ".NS"
            stock_data = get_stock_data(ticker, selected_date, next_date)

            # Generate full date range
            all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))

            if stock_data.empty:
                stock_data = pd.DataFrame(index=all_dates)
                stock_data[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
            else:
                stock_data = stock_data.reindex(all_dates)

            # Merge with numerology
            numerology_subset = numerology_df.set_index('date')
            combined = stock_data.merge(numerology_subset, left_index=True, right_index=True, how='left')
            combined = combined.loc[all_dates]  # ensure consistent order

            # High/Low check
            if combined['High'].notna().any():
                high_val = combined['High'].max()
                low_val = combined['Low'].min()
                st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
            else:
                st.info("No OHLC data available in this period — only numerology shown.")

            combined_reset = combined.reset_index()
            combined_reset.rename(columns={"index": "Date"}, inplace=True)

            # Render table
            html_table = combined_reset.to_html(index=False)
            st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)


    # --- INDEX SECTION ---
    st.subheader("📊 Nifty / BankNifty OHLC + Numerology")

    index_choice = st.radio("Select Index:", ["Nifty 50", "Bank Nifty"])
    index_file = "nifty.xlsx" if index_choice == "Nifty 50" else "banknifty.xlsx"

    index_df = load_excel_data(index_file)
    all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))
    index_range = index_df[(index_df.index >= selected_date) & (index_df.index < next_date)]

    if index_range.empty:
        index_range = pd.DataFrame(index=all_dates)
        index_range[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        index_range = index_range.reindex(all_dates)

    numerology_subset = numerology_df.set_index('date')
    index_combined = index_range.merge(numerology_subset, left_index=True, right_index=True, how='left')
    index_combined = index_combined.loc[all_dates]

    if index_combined['High'].notna().any():
        high_val = index_combined['High'].max()
        low_val = index_combined['Low'].min()
        st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
    else:
        st.info("No index OHLC data available in this period — only numerology shown.")

    index_combined_reset = index_combined.reset_index()
    index_combined_reset.rename(columns={"index": "Date"}, inplace=True)
    html_table = index_combined_reset.to_html(index=False)

    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Mercury":
    st.header("🪐Mercury Phase Analysis")

    # Load mercury data
    mercury_df = pd.read_excel("mercury.xlsx")
    mercury_df['Date'] = pd.to_datetime(mercury_df['Date'], dayfirst=True)
    mercury_df = mercury_df.sort_values('Date')
    # Load moon phase data
    moon_df = pd.read_excel("moon.xlsx")
    moon_df['Date'] = pd.to_datetime(moon_df['Date'], dayfirst=True)

    # Get dates for Amavasya and Poornima
    amavasya_dates = set(moon_df[moon_df['A/P'].str.lower() == "amavasya"]['Date'].dt.date)
    poornima_dates = set(moon_df[moon_df['A/P'].str.lower() == "poornima"]['Date'].dt.date)


    # Load stock symbols from doc.xlsx
    doc_df = pd.read_excel("doc.xlsx")
    available_symbols = sorted(doc_df['Symbol'].dropna().unique().tolist())

    # Load numerology
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], dayfirst=True)

    # mercury Phase & Date
    phase_choice = st.selectbox("Select mercury Phase:", ["Direct", "Retrograde"])
    phase_filtered = mercury_df[mercury_df['D/R'].str.lower() == phase_choice.lower()]
    available_dates = phase_filtered['Date'].dt.strftime("%Y-%m-%d").tolist()
    selected_date_str = st.selectbox(f"Select a {phase_choice} Date:", available_dates)
    selected_date = pd.to_datetime(selected_date_str)

    future_dates = mercury_df[mercury_df['Date'] > selected_date]
    next_date = future_dates.iloc[0]['Date'] if not future_dates.empty else selected_date + pd.Timedelta(days=15)

    # mercury Info
    match = mercury_df[mercury_df['Date'].dt.date == selected_date.date()]
    if match.empty:
        st.error("Selected date not found in mercury data.")
        st.stop()

    # Mercury info at start date
    start_row = mercury_df[mercury_df['Date'].dt.date == selected_date.date()]
    start_degree = start_row.iloc[0]['Degree'] if not start_row.empty else "N/A"
    start_time = start_row.iloc[0]['Time'] if not start_row.empty else "N/A"

    # Mercury info at end date (if exists)
    end_row = mercury_df[mercury_df['Date'].dt.date == next_date.date()]
    end_degree = end_row.iloc[0]['Degree'] if not end_row.empty else "N/A"
    end_time = end_row.iloc[0]['Time'] if not end_row.empty else "N/A"

    # Display both
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Start Date:** {selected_date.date()}  \n**Degree:** {start_degree}  \n**Time:** {start_time}")
    with col2:
        st.markdown(f"**End Date:** {next_date.date()}  \n**Degree:** {end_degree}  \n**Time:** {end_time}")

    

    # Find next mercury date
    future_dates = mercury_df[mercury_df['Date'] > selected_date]
    next_date = future_dates.iloc[0]['Date'] if not future_dates.empty else selected_date + pd.Timedelta(days=15)
    st.markdown(f"### 📅 Period: {selected_date.date()} to {next_date.date()}")

    st.subheader("📈 Symbol OHLC + Numerology")

    # --- SYMBOL SECTION ---
    selected_symbol = st.selectbox("Select Stock Symbol:", available_symbols)

    listing_row = doc_df[doc_df['Symbol'] == selected_symbol]
    if listing_row.empty or pd.isnull(listing_row.iloc[0]['DATE OF INCORPORATION']):
        st.warning("Listing date unavailable.")
    else:
        listing_date = pd.to_datetime(listing_row.iloc[0]['DATE OF INCORPORATION'])

        if selected_date < listing_date:
            st.warning(f"{selected_symbol} was not listed on {selected_date.date()}")
        else:
            ticker = selected_symbol + ".NS"
            stock_data = get_stock_data(ticker, selected_date, next_date)

            # Generate full date range
            all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))

            if stock_data.empty:
                stock_data = pd.DataFrame(index=all_dates)
                stock_data[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
            else:
                stock_data = stock_data.reindex(all_dates)

            # Merge with numerology
            numerology_subset = numerology_df.set_index('date')
            combined = stock_data.merge(numerology_subset, left_index=True, right_index=True, how='left')
            combined = combined.loc[all_dates]  # ensure consistent order

            # High/Low check
            if combined['High'].notna().any():
                high_val = combined['High'].max()
                low_val = combined['Low'].min()
                st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
            else:
                st.info("No OHLC data available in this period — only numerology shown.")

            combined_reset = combined.reset_index()
            combined_reset.rename(columns={"index": "Date"}, inplace=True)

            # Step 7: Highlight rows based on moon phase
            def highlight_moon_rows(row):
                date = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
                if date in amavasya_dates:
                    return ['background-color: #ff2525'] * len(row)  # Light red
                elif date in poornima_dates:
                    return ['background-color: #7aceff'] * len(row)  # Sky blue
                else:
                    return [''] * len(row)

            # Render table
            styled_df = combined_reset.style.apply(highlight_moon_rows, axis=1).format(precision=2)
            html_table = styled_df.to_html()
            st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)


    # --- INDEX SECTION ---
    st.subheader("📊 Nifty / BankNifty OHLC + Numerology")

    index_choice = st.radio("Select Index:", ["Nifty 50", "Bank Nifty"])
    index_file = "nifty.xlsx" if index_choice == "Nifty 50" else "banknifty.xlsx"

    index_df = load_excel_data(index_file)
    all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))
    index_range = index_df[(index_df.index >= selected_date) & (index_df.index < next_date)]

    if index_range.empty:
        index_range = pd.DataFrame(index=all_dates)
        index_range[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        index_range = index_range.reindex(all_dates)

    numerology_subset = numerology_df.set_index('date')
    index_combined = index_range.merge(numerology_subset, left_index=True, right_index=True, how='left')
    index_combined = index_combined.loc[all_dates]

    if index_combined['High'].notna().any():
        high_val = index_combined['High'].max()
        low_val = index_combined['Low'].min()
        st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
    else:
        st.info("No index OHLC data available in this period — only numerology shown.")

    index_combined_reset = index_combined.reset_index()
    index_combined_reset.rename(columns={"index": "Date"}, inplace=True)

    # Step 7: Highlight rows based on moon phase
    def highlight_moon_rows(row):
        date = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
        if date in amavasya_dates:
            return ['background-color: #ff2525'] * len(row)  # Light red
        elif date in poornima_dates:
            return ['background-color: #7aceff'] * len(row)  # Sky blue
        else:
            return [''] * len(row)

    # Step 8: Display styled table
    styled_df = index_combined_reset.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    html_table = styled_df.to_html()
    

    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Sun Number Dates":
    st.title("🌞 Sun Number Dates")

    # Step 1: User selects which date to base SN on
    date_type = st.selectbox("Choose Date Type for Sun Number:", 
                             ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"])

    # Step 2: Map selected date to real SN using numerology_df
    stock_df['Selected Date'] = pd.to_datetime(stock_df[date_type], errors='coerce')
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce')

    # Drop duplicates to ensure clean mapping
    sn_lookup = numerology_df.drop_duplicates(subset='date').set_index('date')['SN']
    stock_df['Selected SN'] = stock_df['Selected Date'].map(sn_lookup)

    # Step 3: Filter by selected SN
    valid_sns = sorted(stock_df['Selected SN'].dropna().unique())
    selected_sn = st.selectbox("Filter by Sun Number:", valid_sns)

    matching_df = stock_df[stock_df['Selected SN'] == selected_sn]

    if matching_df.empty:
        st.warning("No companies found with this SN.")
        st.stop()

    # Step 4: User selects company from filtered list
    company_choice = st.selectbox("Select Company Symbol:", matching_df['Symbol'])
    selected_row = matching_df[matching_df['Symbol'] == company_choice].iloc[0]
    st.markdown(f"**Company Name:** {selected_row['Company Name']}")

    # Step 5: Date input
    default_end = pd.to_datetime("today").normalize()
    default_start = default_end - pd.Timedelta(days=30)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    # Step 6: Get stock data
    ticker = company_choice + ".NS"
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Step 7: Prepare vertical lines based on SN
    sn_vertical_lines = {
        1: ["2025-05-05", "2025-05-07", "2025-05-08", "2025-05-10"],
        2: ["2025-05-03", "2025-05-08", "2025-05-09", "2025-05-13"],
        3: ["2025-05-06", "2025-05-10", "2025-05-11"],
        4: ["2025-05-01", "2025-05-04", "2025-05-11", "2025-05-12"],
        5: ["2025-05-02", "2025-05-05", "2025-05-08", "2025-05-12"],
        6: ["2025-05-01", "2025-05-03", "2025-05-11", "2025-05-13"],
        7: ["2025-05-04", "2025-05-14"],
        8: ["2025-05-05", "2025-05-07", "2025-05-09"],
        9: ["2025-05-02", "2025-05-06", "2025-05-11"]
    }

    vertical_lines = [pd.to_datetime(d) for d in sn_vertical_lines.get(selected_sn, [])]
    vertical_lines = [d for d in vertical_lines if start_date <= d.date() <= end_date]

    # Step 8: Plot candlestick chart
    if not stock_data.empty:
        st.subheader("📈 Candlestick Chart")
        chart = plot_candlestick_chart(stock_data, vertical_lines=vertical_lines)
        st.plotly_chart(chart)
    else:
        st.warning("No stock data found for selected date range.")

    # Step 9: Merge with numerology and show OHLCV + numerology
    st.subheader("📊 OHLC + Numerology Data")
    if not stock_data.empty:
        stock_data.index = pd.to_datetime(stock_data.index)
        numerology_merge = numerology_df.set_index('date')
        merged = stock_data.merge(numerology_merge, left_index=True, right_index=True, how='left').format(precision=2)
        
        # Render as HTML
        html_table = merged.reset_index().to_html()

        # Display
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Panchak":
    st.title("📅 Panchak Dates Analysis")

    # Load Panchak data
    panchak_df = pd.read_excel("panchak.xlsx")
    panchak_df['Start Date'] = pd.to_datetime(panchak_df['Start Date'], errors='coerce', dayfirst=True)
    panchak_df['End Date'] = pd.to_datetime(panchak_df['End Date'], errors='coerce', dayfirst=True)
    panchak_df = panchak_df.dropna(subset=['Start Date', 'End Date']).sort_values('Start Date').reset_index(drop=True)

    # Load moon data
    moon_df = pd.read_excel("moon.xlsx")
    moon_df['Date'] = pd.to_datetime(moon_df['Date'], errors='coerce')
    amavasya_dates = set(moon_df[moon_df['A/P'].str.lower() == "amavasya"]['Date'].dt.date)
    poornima_dates = set(moon_df[moon_df['A/P'].str.lower() == "poornima"]['Date'].dt.date)

    # Symbol list
    symbol_list = ["Nifty", "BankNifty"] + sorted(stock_df['Symbol'].dropna().unique().tolist())
    selected_symbol = st.selectbox("Select Symbol", symbol_list)

    # Select Panchak start date
    selected_start_date = st.selectbox("Select Panchak Start Date:", panchak_df['Start Date'].dt.date.unique())

    # Get the corresponding row
    row = panchak_df[panchak_df['Start Date'].dt.date == selected_start_date].iloc[0]
    start_date = row['Start Date']
    end_date = row['End Date']

    st.markdown(f"### 🕒 Panchak Period: {start_date.date()} to {end_date.date()}")
    st.markdown(f"**Start Time:** {row['Start Time']} | **End Time:** {row['End Time']}")
    st.markdown(f"**Start Degree:** {row['Degree']:.4f}")

    # Helper: load + update index data
    def get_combined_index_data(symbol, start_date, end_date):
        file = "nifty.xlsx" if symbol == "Nifty" else "banknifty.xlsx"
        ticker = "^NSEI" if symbol == "Nifty" else "^NSEBANK"

        df = pd.read_excel(file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        latest_local_date = df.index.max().date()
        if end_date.date() > latest_local_date:
            import yfinance as yf
            fetch_start = latest_local_date + pd.Timedelta(days=1)
            yf_data = yf.download(ticker, start=fetch_start, end=end_date + pd.Timedelta(days=1), progress=False)
            if not yf_data.empty:
                yf_data = yf_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                yf_data.index = pd.to_datetime(yf_data.index)
                df = pd.concat([df, yf_data[~yf_data.index.isin(df.index)]])
                df = df[~df.index.duplicated(keep='last')]

        return df.loc[start_date:end_date - pd.Timedelta(days=1)]

    # Get OHLC data
    if selected_symbol in ["Nifty", "BankNifty"]:
        ohlc = get_combined_index_data(selected_symbol, start_date, end_date)
    else:
        ticker = selected_symbol + ".NS"
        ohlc = get_stock_data(ticker, start_date, end_date)

    # Full date range
    all_dates = pd.date_range(start=start_date, end=end_date)

    if ohlc.empty:
        ohlc = pd.DataFrame(index=all_dates)
        ohlc[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        ohlc = ohlc.reindex(all_dates)

    # Load and prepare numerology data
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce').dt.date
    numerology_subset = numerology_df.set_index('date')

    # Merge OHLC and numerology
    merged = ohlc.merge(numerology_subset, left_index=True, right_index=True, how='left')
    merged = merged.loc[all_dates]
    merged = merged.reset_index().rename(columns={"index": "Date"})

    # High/Low display
    if merged['High'].notna().any():
        high_val = merged['High'].max()
        low_val = merged['Low'].min()
        st.markdown(f"**📈 High:** {high_val:.2f} | 📉 Low:** {low_val:.2f}")
    else:
        st.warning("⚠ No OHLC data available for this period — only numerology is shown.")

    # Highlight Amavasya / Poornima
    def highlight_moon_rows(row):
        date = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
        if date in amavasya_dates:
            return ['background-color: #ffcccc'] * len(row)  # Light red
        elif date in poornima_dates:
            return ['background-color: #ccf2ff'] * len(row)  # Sky blue
        else:
            return [''] * len(row)

    # Display styled table
    styled_df = merged.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    html_table = styled_df.to_html()
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    # --- 📊 Post-Panchak Period ---
    st.subheader("📊 Post-Panchak Period Analysis")

    # Find next Panchak start date
    future_rows = panchak_df[panchak_df['Start Date'] > end_date]
    if not future_rows.empty:
        next_start_date = future_rows.iloc[0]['Start Date']
    else:
        next_start_date = end_date + pd.Timedelta(days=10)

    post_start_date = end_date + pd.Timedelta(days=1)
    post_end_date = next_start_date

    st.markdown(f"### ⏭️ Period: {post_start_date.date()} to {(post_end_date).date()}")


    # Get OHLC for post-Panchak period
    if selected_symbol in ["Nifty", "BankNifty"]:
        post_ohlc = get_combined_index_data(selected_symbol, post_start_date, post_end_date)
    else:
        ticker = selected_symbol + ".NS"
        post_ohlc = get_stock_data(ticker, post_start_date, post_end_date)

    # Full date range
    post_dates = pd.date_range(start=post_start_date, end=post_end_date - pd.Timedelta(days=1))

    if post_ohlc.empty:
        post_ohlc = pd.DataFrame(index=post_dates)
        post_ohlc[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        post_ohlc = post_ohlc.reindex(post_dates)

    # Merge with numerology
    post_merged = post_ohlc.merge(numerology_subset, left_index=True, right_index=True, how='left')
    post_merged = post_merged.loc[post_dates]
    post_merged = post_merged.reset_index().rename(columns={"index": "Date"})

    # High/Low display
    if post_merged['High'].notna().any():
        high_val = post_merged['High'].max()
        low_val = post_merged['Low'].min()
        st.markdown(f"**📈 High:** {high_val:.2f} | 📉 Low:** {low_val:.2f}")
    else:
        st.info("⚠ No OHLC data for post-Panchak period — only numerology shown.")

    # Highlight moon phases
    styled_post_df = post_merged.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    post_html_table = styled_post_df.to_html()
    st.markdown(f'<div class="scroll-table">{post_html_table}</div>', unsafe_allow_html=True)

