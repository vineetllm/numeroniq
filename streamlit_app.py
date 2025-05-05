import streamlit as st
import pandas as pd
import yfinance as yf
import re
import plotly.graph_objects as go
from datetime import datetime

# Set wide layout
st.set_page_config(page_title="Stock Screener - IPO & Numerology", layout="wide")

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
    stock_data = yf.download(ticker, start=start_date, end=end_date)
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

char_to_num = {letter: num for num, letters in chaldean_map.items() for letter in letters.split()}

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
    reduced_values = []

    for word in words:
        word_val = get_word_value(word)
        reduced_val = reduce_to_single_digit(word_val)
        word_parts.append(f"{word_val}({reduced_val})")
        reduced_values.append(reduced_val)

    if not reduced_values:
        return None, None  

    total_sum = sum(reduced_values)
    final_val = reduce_to_single_digit(total_sum)

    equation = f"{' + '.join(word_parts)} = ({total_sum}){final_val}"
    return final_val, equation



st.title("📊 Stock Screener - IPO & Numerology Insight")

# === Toggle between filtering methods ===
filter_mode = st.radio("Choose Filter Mode:", ["Home", "Filter by Sector/Symbol", "Filter by Numerology","Name Numerology (Company Name)", "View Nifty/BankNifty OHLC"])

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
        st.dataframe(display_cols, use_container_width=True)

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

                st.dataframe(all_numerology_df, use_container_width=True, hide_index=True)
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
                            
                        st.dataframe(matched_numerology, use_container_width=True)
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
                    st.dataframe(all_numerology_df[all_cols], use_container_width=True, hide_index=True)
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


        st.dataframe(matching_stocks[cols_order], use_container_width=True)

    else:
        st.info("No companies found with matching numerology dates.")

elif filter_mode == "Name Numerology (Company Name)":
    st.subheader("🔢 Chaldean Name Numerology for Company Names")
    
    use_ltd = st.radio(
        "For company names that contain 'Ltd' or 'Limited', include it in numerology calculation?",
        ["Yes", "No"],
        index=1
    )


    numerology_data = []
    for _, row in stock_df.iterrows():
        company_original = row['Company Name']
        symbol = str(row['Symbol'])

        # Remove 'Ltd' or 'Limited' if user chose "No"
        if use_ltd == "No":
            company_clean = re.sub(r'\b(Ltd|Limited)\b', '', company_original, flags=re.IGNORECASE).strip()
        else:
            company_clean = company_original

        # Calculate numerology
        company_num, company_eq = calculate_numerology(company_clean)
        symbol_num, symbol_eq = calculate_numerology(symbol)

        numerology_data.append({
            'Symbol': row['Symbol'],
            'Company Name': company_original,
            'Company Numerology Final': company_num,
            'Numerology Equation (Company Name)': company_eq,
            'Symbol Numerology Final': symbol_num,
            'Numerology Equation (Symbol)': symbol_eq
        })

    numerology_df_display = pd.DataFrame(numerology_data)

    # === Filters ===
    col1, col2 = st.columns(2)

    with col1:
        company_filter = st.selectbox(
            "Select Company (or choose All)",
            options=["All"] + sorted(numerology_df_display['Company Name'].unique())
        )

    with col2:
        number_filter = st.selectbox(
            "Select Final Numerology Number",
            options=["All"] + sorted(numerology_df_display['Company Numerology Final'].unique())
        )

    filtered_df = numerology_df_display.copy()

    if company_filter != "All":
        filtered_df = filtered_df[filtered_df['Company Name'] == company_filter]

    if number_filter != "All":
        filtered_df = filtered_df[filtered_df['Company Numerology Final'] == number_filter]
    
    filtered_df = filtered_df.drop(columns=['Company Numerology Final', 'Symbol Numerology Final'], errors='ignore')


    # === Display Filtered Table ===
    st.dataframe(filtered_df, use_container_width=True)

elif filter_mode == "Home":
    st.title("🏠 Company Snapshot")

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

            company_name_original = str(row['Company Name'])
            symbol_name = str(row['Symbol'])

            if use_ltd_home == "No":
                company_clean = re.sub(r'\b(Ltd|Limited)\b', '', company_name_original, flags=re.IGNORECASE).strip()
            else:
                company_clean = company_name_original

            company_num, company_eq = calculate_numerology(company_clean)
            symbol_num, symbol_eq = calculate_numerology(symbol_name)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Company Name Equation:** {company_eq}")
            with col2:
                st.markdown(f"**Symbol Equation:** {symbol_eq}")


            # --- Line 5: Zodiac Signs ---
            st.markdown("### ♈ Zodiac Information (Based on Dates)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**DOC Zodiac Sign:** {row.get('DOC zodiac sign', 'N/A')}")
                st.markdown(f"**DOC Zodiac Number:** {row.get('DOC zodiac number', 'N/A')}")

            with col2:
                st.markdown(f"**NSE Zodiac Sign:** {row.get('NSE zodiac sign', 'N/A')}")
                st.markdown(f"**NSE Zodiac Number:** {row.get('NSE zodiac number', 'N/A')}")

            with col3:
                st.markdown(f"**BSE Zodiac Sign:** {row.get('BSE zodiac sign', 'N/A')}")
                st.markdown(f"**BSE Zodiac Number:** {row.get('BSE zodiac number', 'N/A')}")

            # --- Candlestick Chart (After Zodiac Info) ---
            st.markdown("### 📈 Stock Price Candlestick Chart")

            # Add start and end date selectors for the user to filter the data range
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))

            ticker = row['Symbol']  # Get the company's symbol

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
    st.dataframe(filtered_merged_reset[existing_cols], use_container_width=True, hide_index=True)



    if st.checkbox("📊 Show Closing Price Chart"):
        st.line_chart(filtered_data['Close'])
    









