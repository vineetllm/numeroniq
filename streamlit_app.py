import streamlit as st
import pandas as pd
import yfinance as yf
import re
import plotly.graph_objects as go
from datetime import datetime
import base64

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



# Load stock data
@st.cache_data
def load_stock_data():
    df = pd.read_excel("doc.xlsx")
    df['NSE LISTING DATE'] = pd.to_datetime(df['NSE LISTING DATE'], errors='coerce')
    df['BSE LISTING DATE'] = pd.to_datetime(df['BSE LISTING DATE'], errors='coerce')
    df['DATE OF INCORPORATION'] = pd.to_datetime(df['DATE OF INCORPORATION'], errors='coerce')
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

def plot_candlestick_chart(stock_data):
    """
    Generate and return a candlestick chart using Plotly.
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

import re

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
filter_mode = st.radio("Choose Filter Mode:", ["Company Overview", "Filter by Sector/Symbol", "Filter by Numerology","Name Numerology", "View Nifty/BankNifty OHLC"])

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
        st.dataframe(display_cols, use_container_width=True, selection_mode='single-row')

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

                st.dataframe(all_numerology_df, use_container_width=True, hide_index=True, selection_mode='single-row')
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
                            
                        st.table(matched_numerology)
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
                    st.table(all_numerology_df[all_cols])
                else:
                    st.warning("No numerology data found for selected date field across these companies.")

            else:
                # Multiple companies manually selected but show_all_in_sector is False
                st.info("Select a single symbol (uncheck the box) to see numerology data.")

    else:
        st.warning("No matching data found.")


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


        st.table(matching_stocks[cols_order])

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


    # === Display Filtered Table ===
    st.table(filtered_df)

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
                index=0,
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

            if 'DN (Formatted)' not in numerology_df.columns:
                dn_values = numerology_df['date'].apply(calculate_destiny_number)
                numerology_df['DN Raw'] = dn_values.apply(lambda x: x[0])
                numerology_df['DN'] = dn_values.apply(lambda x: x[1])
                numerology_df['DN (Formatted)'] = numerology_df.apply(
                    lambda row: f"({row['DN Raw']}){row['DN']}" if pd.notnull(row['DN Raw']) else None,
                axis=1
            )


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
                else:
                    st.info(f"No numerology data available for {dt_type}.")
            else:
                st.info(f"No date available for {dt_type}.")



            # --- Candlestick Chart (After Zodiac Info) ---
            st.markdown("### 📈 Stock Price Candlestick Chart")

            # Add start and end date selectors for the user to filter the data range
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))

            ticker = str(row['Symbol']).upper() + ".NS"  # Get the company's symbol

            if start_date and end_date:
                stock_data = get_stock_data(ticker, start_date, end_date)
                if not stock_data.empty:
                    chart = plot_candlestick_chart(stock_data)
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

    # Load data from file
    excel_data = load_excel_data(file)
    last_excel_date = excel_data.index[-1].date()

    # Fetch data from yfinance only after last_excel_date
    @st.cache_data(ttl=3600)
    def fetch_yfinance_data(symbol, start_date):
        yf_data = yf.download(symbol, start=start_date, interval="1d")[['Open', 'High', 'Low', 'Close']]
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

    # Reorder columns: Volatility % and Close % first
    reordered_cols = ['Volatility %', 'Close %', 'Open', 'High', 'Low', 'Close']
    full_data = full_data[reordered_cols]

    # Filters in horizontal layout
    st.markdown("### 🔍 Filter OHLC Table")
    col1, col2, col3 = st.columns(3)

    with col1:
        vol_op = st.selectbox("Volatility Operator", ["All", "<", "<=", ">", ">=", "=="])
        vol_val = st.number_input("Volatility % Value", value=2.0, step=0.1)

    with col2:
        close_op = st.selectbox("Close % Operator", ["All", "<", "<=", ">", ">=", "=="])
        close_val = st.number_input("Close % Value", value=0.5, step=0.1)

    filtered_data = full_data.copy()

    if vol_op != "All":
        filtered_data = filtered_data.query(f"`Volatility %` {vol_op} @vol_val")

    if close_op != "All":
        filtered_data = filtered_data.query(f"`Close %` {close_op} @close_val")

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
        dn_filter = st.selectbox("DN", ["All"] + sorted(numerology_df['DN'].dropna().unique()))

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
        filtered_merged = filtered_merged[filtered_merged['DN'] == dn_filter]
    if sn_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['SN'] == sn_filter]
    if hp_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['HP'] == hp_filter]
    if dayn_filter != "All":
        filtered_merged = filtered_merged[filtered_merged['Day Number'] == dayn_filter]

    # Display filtered, aligned data
    st.markdown("### 🔢 OHLC + Numerology Alignment")
    # Reorder columns
    ordered_cols = ['Volatility %', 'Close %', 'Open', 'High', 'Low', 'Close']
    numerology_cols = ['BN', 'DN', 'SN', 'HP', 'Day Number', 'BN Planet','DN Planet', 'SN Planet', 'HP Planet', 'Day Number Planet']
    # Include date as a column if it's not already (currently index)
    filtered_merged_reset = filtered_merged.reset_index()

    # Final column order
    final_order = ['Date'] + numerology_cols + ordered_cols
    existing_cols = [col for col in final_order if col in filtered_merged_reset.columns]

    # Desired column order (adjust as needed if columns vary)
    desired_order = [
        'Date',
        'BN', 'DN', 
        'SN', 'HP', 
        'Day Number', 'BN Planet',
        'DN Planet',  'SN Planet',
        'HP Planet', 'Day Number Planet',
        'Volatility %', 'Close %',
        'Open', 'High',
        'Low', 'Close'
    ]
    
    # Display reordered table
    st.table(filtered_merged_reset[existing_cols])



    if st.checkbox("📊 Show Closing Price Chart"):
        st.line_chart(filtered_data['Close'])
    





