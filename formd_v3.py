import requests
import pandas as pd
import streamlit as st
from openai import OpenAI
import xml.etree.ElementTree as ET
import re
import time
from datetime import datetime
import pytz
from urllib.parse import quote


def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')

# =========================
# Constants
# =========================
URL = "https://efts.sec.gov/LATEST/search-index"

# Use a descriptive UA per SEC guidance (name + email). Avoid browser-y UAs here.
YOUR_NAME = "Fund Launch Partners Acquisitions"
YOUR_EMAIL = "acquisitions@fundlaunchpartners.com"

HEADERS = {
    "User-Agent": f"{YOUR_NAME} FormD Scraper ({YOUR_EMAIL})",
    "Accept": "application/json"
}

DOC_HEADERS = {
    "User-Agent": f"{YOUR_NAME} FormD Scraper ({YOUR_EMAIL})",
    "Accept-Encoding": "gzip, deflate",
}

# =========================
# Sidebar: simple chatbot UI
# =========================
with st.sidebar:
    with st.popover("OpenAI API Key", help="Please insert you API Key to use the chatbot"):
        st.header("💬 ChatGPT")
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = [{"role": "assistant", "content": "Ask me anything about the Form D filings"}]

    for msg in st.session_state["chatbot_messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)
        st.session_state["chatbot_messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model="gpt-4o", messages=st.session_state["chatbot_messages"])
        msg = response.choices[0].message.content
        st.session_state["chatbot_messages"].append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

# =========================
# Helper Functions
# =========================
def format_date(date_str):
    """Format date string to 'MM/DD/YYYY' format."""
    if not date_str or date_str in ["Yet to Occur", "Unknown"]:
        return date_str
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%m/%d/%Y")
    except:
        return date_str


def detect_fund_stage(fund_name):
    """
    Detect fund stage (I, II, III, IV) from fund name.
    Returns detected stage or "N/A" if not found.
    Filters out funds V+ (5 and above).
    """
    if not fund_name:
        return "N/A"
    
    # Common patterns for fund stages
    patterns = [
        r'\bFund\s+([IVX]+)\b',  # "Fund I", "Fund II", "Fund III"
        r'\b([IVX]+)\s+Fund\b',  # "I Fund", "II Fund"
        r'\bLP\s+([IVX]+)\b',    # "LP I", "LP II"
        r'\b([IVX]+)\s+LP\b',    # "I LP", "II LP"
        r'\bL\.P\.\s+([IVX]+)\b', # "L.P. I"
        r'\b([IVX]+)\s+L\.P\.\b', # "I L.P."
        r'\bFund\s+(\d+)\b',     # "Fund 1", "Fund 2"
        r'\b(\d+)\s+Fund\b',     # "1 Fund", "2 Fund"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, fund_name, re.IGNORECASE)
        if match:
            stage = match.group(1).upper()
            # Convert roman numerals and numbers to standard format
            roman_map = {
                'I': 'I', 'II': 'II', 'III': 'III', 'IV': 'IV',
                '1': 'I', '2': 'II', '3': 'III', '4': 'IV'
            }
            
            # Check if it's a number
            if stage.isdigit():
                num = int(stage)
                if num > 4:
                    return "V+"  # Mark funds beyond IV
                return roman_map.get(stage, "N/A")
            
            # Check if it's roman numerals beyond IV
            if stage not in roman_map:
                return "V+"
                
            return roman_map.get(stage, stage)
    
    return "N/A"


def detect_asset_class(fund_name, investment_subtype):
    """
    Detect detailed asset class from fund name when SEC says "Other Investment Fund".
    Returns enhanced category or original subtype.
    """
    if not fund_name:
        return investment_subtype
    
    fund_name_lower = fund_name.lower()
    
    # Private Credit/Debt keywords
    credit_keywords = ['credit', 'debt', 'lending', 'loan', 'fixed income', 'income fund',
                      'direct lending', 'senior debt', 'subordinated', 'mezzanine',
                      'structured credit', 'distressed debt', 'asset-based lending',
                      'specialty finance', 'private debt', 'middle market lending',
                      'unitranche', 'bdc', 'clo']
    
    # Infrastructure keywords
    infra_keywords = ['infrastructure', 'energy transition', 'renewable', 'utilities']
    
    # Real Assets keywords
    real_assets_keywords = ['real assets', 'natural resources', 'commodities', 'timber', 'farmland']
    
    # Private Equity sub-categories
    buyout_keywords = ['buyout', 'buy-out', 'acquisition', 'control equity']
    growth_keywords = ['growth equity', 'growth capital', 'expansion capital']
    
    # Only add (OIF) notation if SEC classified it as "Other Investment Fund"
    if investment_subtype == "Other Investment Fund":
        if any(kw in fund_name_lower for kw in credit_keywords):
            return "Private Credit (OIF)"
        elif any(kw in fund_name_lower for kw in infra_keywords):
            return "Infrastructure (OIF)"
        elif any(kw in fund_name_lower for kw in real_assets_keywords):
            return "Real Assets (OIF)"
        elif any(kw in fund_name_lower for kw in buyout_keywords):
            return "Buyout PE (OIF)"
        elif any(kw in fund_name_lower for kw in growth_keywords):
            return "Growth Equity (OIF)"
    
    # Return original subtype if we couldn't detect or if it's not OIF
    return investment_subtype


def calculate_percent_raised(total_offering, total_sold):
    """Calculate percentage of offering that has been raised."""
    try:
        offering = float(total_offering) if total_offering else 0
        sold = float(total_sold) if total_sold else 0
        
        if offering > 0:
            percent = (sold / offering) * 100
            return f"{percent:.1f}%"
        return "N/A"
    except:
        return "N/A"


def format_currency(amount):
    """Format currency amounts for display."""
    try:
        num = float(amount)
        if num >= 1_000_000:
            return f"${num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"${num/1_000:.1f}K"
        else:
            return f"${num:.0f}"
    except:
        return amount if amount else "N/A"

# =========================
# Data fetch & parsing
# =========================
def _request_search(params, retries=2, timeout=30):
    """Internal: call SEC search API with light retry/backoff for 429s."""
    for attempt in range(retries + 1):
        resp = requests.get(URL, headers=HEADERS, params=params, timeout=timeout)
        if resp.status_code == 429 and attempt < retries:
            retry_after = min(int(resp.headers.get("Retry-After", "1")), 5)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        return resp.json()
    return None


def fetch_sec_filings(start_date, end_date, page_size=200, max_pages=150):
    """
    Fetch SEC Form D filings between two dates, across pages.
    Includes a compatibility fallback if a paged call returns empty.
    """
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)

    if start_str > end_str:
        st.warning("Start Date is after End Date.")
        return []

    base_params = {
        "dateRange": "custom",
        "startdt": start_str,
        "enddt": end_str,
        "forms": "D",
    }

    all_results = []
    pages = 0
    offset = 0
    
    while pages < max_pages:
        params = dict(base_params)
        params.update({
            "from": offset,
            "size": page_size,
            "sort": "file_date:desc"
        })

        try:
            data = _request_search(params)
        except requests.RequestException as e:
            st.error(f"Request failed at offset {offset}: {e}")
            break
        if not data:
            break

        page_results = clean_up(data)
        total_field = data.get("hits", {}).get("total", 0)
        total = total_field.get("value", 0) if isinstance(total_field, dict) else int(total_field or 0)

        if not page_results:
            if offset == 0:
                fallback_params = dict(base_params)
                try:
                    fallback_data = _request_search(fallback_params)
                except requests.RequestException as e:
                    st.error(f"Fallback request failed: {e}")
                    break
                if not fallback_data:
                    break

                first_results = clean_up(fallback_data)
                if not first_results:
                    break
                all_results.extend(first_results)

                offset = len(first_results)
                total_field_fb = fallback_data.get("hits", {}).get("total", 0)
                total_fb = total_field_fb.get("value", 0) if isinstance(total_field_fb, dict) else int(total_field_fb or 0)
                total = max(total, total_fb)

                if offset >= total:
                    break

                pages += 1
                continue
            else:
                break

        all_results.extend(page_results)
        offset += page_size
        pages += 1

        if offset >= total:
            break

    return all_results


def clean_up(response):
    """Extract and format data from SEC response."""
    results = []
    for item in response.get("hits", {}).get("hits", []):
        source = item.get("_source", {})
        accession_number = source.get("adsh", "")
        primary_doc = source.get("primary_document", f"{accession_number}.txt")
        ciks = source.get("ciks", [])
        cik = ciks[0] if ciks else ""
        edgar_link = f'<a href="https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number.replace("-", "")}/{primary_doc}" target="_blank">Form D</a>'

        result = {
            "Company Name": ', '.join(source.get("display_names", [])),
            "File Date": source.get("file_date", ""),
            "Business Location(s)": ', '.join(source.get("biz_locations", [])),
            "CIK": cik,
            "Accession Number": accession_number,
            "Primary Document": primary_doc,
            "Form D Filing": edgar_link
        }
        results.append(result)
    return results


def get_sales_compensation(root):
    """Extract sales compensation recipients from Form D XML."""
    recipients = []
    for recipient in root.findall(".//offeringData/salesCompensationList/recipient"):
        name_elem = recipient.find("recipientName")
        if name_elem is not None and name_elem.text:
            recipients.append(name_elem.text.strip())
    return "; ".join(recipients) if recipients else ""


def get_formd_details(
    formd_url,
    required_industry_group="Pooled Investment Fund",
    required_subtype=None,
    allowed_years=None,
    allowed_stages=None,
    min_fund_size=0,
    max_fund_size=300_000_000
):
    """Parse primary_doc.xml and filter by industry subtype, year, fund stage, and fund size."""
    try:
        response = requests.get(formd_url, headers=DOC_HEADERS, timeout=30)
        response.raise_for_status()
        content = response.text

        xml_match = re.search(r"<XML>(.*?)</XML>", content, re.DOTALL | re.IGNORECASE)
        if not xml_match:
            return None

        xml_content = xml_match.group(1).strip()
        xml_content = re.sub(r"[^\x09\x0A\x0D\x20-\x7F]", "", xml_content)

        root = ET.fromstring(xml_content)
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        def get(path):
            el = root.find(path)
            return el.text.strip() if el is not None and el.text else ""

        # Year of Inc. filter
        year_elem = root.find(".//primaryIssuer/yearOfInc/value")
        if allowed_years:
            if year_elem is None or (year_elem.text not in allowed_years):
                return None
            year_text = year_elem.text
        else:
            year_text = year_elem.text if year_elem is not None else ""

        # Industry group & subtype
        industry_group = get(".//offeringData/industryGroup/industryGroupType")
        
        if industry_group != "Pooled Investment Fund":
            return None
        
        investment_subtype = get(".//offeringData/industryGroup/investmentFundInfo/investmentFundType")
        
        if required_industry_group and industry_group != required_industry_group:
            return None
        
        # Apply subtype filter
        if required_subtype:
            if required_subtype == "Other Investment Fund":
                if investment_subtype != "Other Investment Fund":
                    return None
            elif investment_subtype != required_subtype:
                return None

        # Fund size filter
        total_offering = get(".//offeringData/offeringSalesAmounts/totalOfferingAmount")
        try:
            offering_amount = float(total_offering) if total_offering else 0
            if offering_amount < min_fund_size or offering_amount > max_fund_size:
                return None
        except:
            pass

        entity_name = get(".//primaryIssuer/entityName")
        
        # Detect fund stage
        fund_stage = detect_fund_stage(entity_name)
        
        # Fund stage filter
        if allowed_stages and fund_stage not in allowed_stages:
            return None
        
        # Detect enhanced asset class
        enhanced_type = detect_asset_class(entity_name, investment_subtype)

        # Financial data
        total_sold = get(".//offeringData/offeringSalesAmounts/totalAmountSold")
        total_remaining = get(".//offeringData/offeringSalesAmounts/totalRemaining")
        
        # Calculate % raised
        percent_raised = calculate_percent_raised(total_offering, total_sold)

        # Date of first sale
        sale_date_val = root.find(".//offeringData/typeOfFiling/dateOfFirstSale/value")
        yet_to_occur = root.find(".//offeringData/typeOfFiling/dateOfFirstSale/yetToOccur")
        
        if yet_to_occur is not None:
            date_first_sale = "Yet to Occur"
        elif sale_date_val is not None and (sale_date_val.text or "").strip():
            date_first_sale = sale_date_val.text.strip()
        else:
            date_first_sale = "Unknown"

        # Sales compensation / Placement agent detection
        sales_comp = get_sales_compensation(root)
        using_placement_agent = "Yes" if sales_comp else "No"

        city = get(".//primaryIssuer/issuerAddress/city")
        state = get(".//primaryIssuer/issuerAddress/stateOrCountry")
        location = f"{city}, {state}" if city and state else (city or state or "N/A")

        data = {
            # Main view columns (in order)
            "Fund Name": entity_name,
            "Fund Stage": fund_stage,
            "Investment Type": enhanced_type,
            "Fund Size": format_currency(total_offering),
            "Amount Raised": format_currency(total_sold),
            "% Raised": percent_raised,
            "Date of First Sale": format_date(date_first_sale),
            "Year of Incorporation": year_text,
            "Location": location,
            
            # Expandable/Detail columns
            "Name of Signer": get(".//offeringData/signatureBlock/signature/nameOfSigner"),
            "Title": get(".//offeringData/signatureBlock/signature/signatureTitle"),
            "Phone Number": get(".//primaryIssuer/issuerPhoneNumber"),
            "Street": get(".//primaryIssuer/issuerAddress/street1"),
            "City": city,
            "State": state,
            "Zip": get(".//primaryIssuer/issuerAddress/zipCode"),
            "Total Investors": get(".//offeringData/investors/totalNumberAlreadyInvested"),
            "Minimum Investment": format_currency(get(".//offeringData/minimumInvestmentAccepted")),
            "Total Remaining": format_currency(total_remaining),
            "Issuer Size": get(".//offeringData/issuerSize/revenueRange") or get(".//offeringData/issuerSize/aggregateNetAssetValueRange"),
            "Using Placement Agent": using_placement_agent,
            "Sales Compensation": sales_comp,
            "Federal Exemptions": get(".//offeringData/federalExemptionsExclusions/item"),
            
            # Raw values for CSV and calculations
            "Total Offering Amount (Raw)": total_offering,
            "Total Amount Sold (Raw)": total_sold,
            "Date of First Sale (Raw)": date_first_sale,
        }

        return data

    except Exception as e:
        st.warning(f"⚠️ Error parsing filing: {e}")
        return None


def create_main_view_df(detailed_data):
    """Create DataFrame with only main view columns in the correct order."""
    main_columns = [
        "Fund Name",
        "Fund Stage",
        "Fund Size",
        "Amount Raised",
        "% Raised",
        "Investment Type",
        "Year of Incorporation",
        "Date of First Sale",
        "Location"
    ]
    
    df = pd.DataFrame(detailed_data)
    
    # Sort by Date of First Sale (newest first, "Yet to Occur" after dates)
    # Convert raw date for sorting
    def parse_date_for_sort(date_str):
        if date_str == "Yet to Occur":
            return datetime(1900, 1, 2)  # Put "Yet to Occur" after dates but before Unknown
        elif date_str == "Unknown":
            return datetime(1900, 1, 1)  # Put "Unknown" at bottom
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return datetime(1900, 1, 1)
    
    df['_sort_date'] = df['Date of First Sale (Raw)'].apply(parse_date_for_sort)
    df = df.sort_values('_sort_date', ascending=False)
    df = df.drop('_sort_date', axis=1)
    df = df.reset_index(drop=True)  # Remove index column
    
    return df[main_columns]


def create_expandable_section(row):
    """Create expandable section for a single row with details."""
    # Create SEC Form D link for fund name
    fund_name = row.get('Fund Name', '')
    cik = row.get('CIK', '')
    acc = row.get('Accession Number', '').replace("-", "")
    primary_doc = row.get('Primary Document', f"{acc}.txt")
    sec_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary_doc}"
    
    # Build clean text output
    details = f"""
**{fund_name}**  
[View SEC Form D Filing]({sec_url})

---

**👤 KEY PEOPLE**

**Name of Signer:** {row.get('Name of Signer', 'N/A')}  
**Title:** {row.get('Title', 'N/A')}

**📞 CONTACT & OUTREACH**

**Phone Number:** {row.get('Phone Number', 'N/A')}  
**Street:** {row.get('Street', 'N/A')}  
**City:** {row.get('City', 'N/A')}  
**State:** {row.get('State', 'N/A')}  
**Zip:** {row.get('Zip', 'N/A')}  
**Form D Filing:** {row.get('Form D Filing', 'N/A')}

**💰 FINANCIAL DETAILS**

**Total Investors:** {row.get('Total Investors', 'N/A')}  
**Minimum Investment:** {row.get('Minimum Investment', 'N/A')}  
**Total Remaining:** {row.get('Total Remaining', 'N/A')}  
**Issuer Size:** {row.get('Issuer Size', 'N/A')}

**🤝 DEAL STRUCTURE**

**Using Placement Agent:** {row.get('Using Placement Agent', 'N/A')}  
**Sales Compensation:** {row.get('Sales Compensation', 'N/A')}  
**Federal Exemptions:** {row.get('Federal Exemptions', 'N/A')}

**🔍 TECHNICAL/REFERENCE**

**CIK:** {row.get('CIK', 'N/A')}  
**Accession Number:** {row.get('Accession Number', 'N/A')}  
**Primary Document:** {row.get('Primary Document', 'N/A')}
"""
    return details


# =========================
# Streamlit App
# =========================
def main():
    # Display logo at top left
    try:
        st.image("FLP_Logo_v3.png", width=300)
    except:
        pass  # If logo not found, continue without it
    
    st.title('Emerging Manager Sourcing (Form D)')
    
    # User-Agent collapsible info
    with st.expander("ℹ️ SEC Compliance Information"):
        st.write(f"""
        **User-Agent:** {YOUR_NAME} ({YOUR_EMAIL})
        
        This identifier is sent to the SEC when we access Form D filings, as required by SEC guidelines.
        """)
    st.markdown("---")

    if "filing_results" not in st.session_state:
        st.session_state["filing_results"] = None
    if "last_search_time" not in st.session_state:
        st.session_state["last_search_time"] = None

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")

    # Fund size filter
    col3, col4 = st.columns(2)
    with col3:
        min_size = st.number_input("Minimum Fund Size ($M)", min_value=0, max_value=1000, value=0, step=10)
    with col4:
        max_size = st.number_input("Maximum Fund Size ($M)", min_value=0, max_value=1000, value=300, step=10)

    # Fund Stage filter (multi-select checkboxes)
    st.write("**Fund Stage:**")
    col_stage1, col_stage2, col_stage3, col_stage4, col_stage5, col_stage6 = st.columns(6)
    with col_stage1:
        stage_na = st.checkbox("N/A (Unknown)", value=True, key="stage_na")
    with col_stage2:
        stage_i = st.checkbox("Fund I", value=True, key="stage_i")
    with col_stage3:
        stage_ii = st.checkbox("Fund II", value=True, key="stage_ii")
    with col_stage4:
        stage_iii = st.checkbox("Fund III", value=True, key="stage_iii")
    with col_stage5:
        stage_iv = st.checkbox("Fund IV", value=False, key="stage_iv")
    with col_stage6:
        stage_v_plus = st.checkbox("Fund V+", value=False, key="stage_v_plus")
    
    st.caption("Note: N/A includes funds where stage couldn't be detected from name (often Fund I)")
    
    # Utah-based filter
    utah_only = st.checkbox("Utah-based funds", value=False, key="utah_only")
    
    # Build allowed stages list
    allowed_stages = []
    if stage_na:
        allowed_stages.append("N/A")
    if stage_i:
        allowed_stages.append("I")
    if stage_ii:
        allowed_stages.append("II")
    if stage_iii:
        allowed_stages.append("III")
    if stage_iv:
        allowed_stages.append("IV")
    if stage_v_plus:
        allowed_stages.append("V+")

    # Dynamic year filter
    current_year = datetime.now().year
    available_years = [str(y) for y in range(2020, current_year + 2)]
    selected_years = st.multiselect(
        "Year of Incorporation",
        options=available_years,
        default=[],
        help="Select years to filter by, or leave empty for all years"
    )

    industry_group_fixed = "Pooled Investment Fund"
    
    # Fund Type dropdown
    industry_subtype = st.selectbox(
        "Fund Type",
        ["Any", "Private Equity Fund", "Hedge Fund", "Venture Capital Fund", "Private Credit (Keyword Detection)", "Other Investment Fund"],
        index=1
    )

    # Buttons row
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        search_button = st.button("🔍 Search & Filter", type="primary")
    with col_btn2:
        clear_button = st.button("🔄 Clear Filters")
    
    # Clear filters functionality
    if clear_button:
        st.session_state.clear()
        st.rerun()

    # Combined search and filter
    if search_button:
        # Fetch filings
        with st.spinner("Fetching Form D filings..."):
            filings = fetch_sec_filings(start_date, end_date, page_size=200, max_pages=150)
            
            if filings:
                total_fetched = len(filings)
                st.info(f"📥 Searched {total_fetched:,} total filings from {start_date} to {end_date}")
                
                detailed_data = []

                # Map "Private Credit (Keyword Detection)" to "Other Investment Fund" for SEC filter
                if industry_subtype == "Any":
                    subtype_filter = None
                elif industry_subtype == "Private Credit (Keyword Detection)":
                    subtype_filter = "Other Investment Fund"
                else:
                    subtype_filter = industry_subtype

                # Convert selected years to tuple or None
                allowed_years = tuple(selected_years) if selected_years else None
                
                # Convert fund sizes to dollars
                min_fund_size = min_size * 1_000_000
                max_fund_size = max_size * 1_000_000

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_filings = len(filings)
                
                for idx, row in enumerate(filings):
                    # Update progress
                    progress = (idx + 1) / total_filings
                    progress_bar.progress(progress)
                    status_text.text(f"Processing filing {idx + 1} of {total_filings}...")
                    
                    cik = row.get("CIK", "")
                    if not cik:
                        continue
                    acc = (row.get("Accession Number", "") or "").replace("-", "")
                    primary_doc = row.get("Primary Document") or f"{acc}.txt"
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary_doc}"

                    filing_data = get_formd_details(
                        url,
                        required_industry_group=industry_group_fixed,
                        required_subtype=subtype_filter,
                        allowed_years=allowed_years,
                        allowed_stages=allowed_stages if allowed_stages else None,
                        min_fund_size=min_fund_size,
                        max_fund_size=max_fund_size
                    )
                    if filing_data:
                        # Apply Utah filter if enabled
                        if utah_only:
                            location = filing_data.get("Location", "")
                            if "UT" not in location.upper() and "UTAH" not in location.upper():
                                continue  # Skip non-Utah funds
                        
                        # Add technical fields for expandable section
                        filing_data["CIK"] = cik
                        filing_data["Accession Number"] = row.get("Accession Number", "")
                        filing_data["Primary Document"] = primary_doc
                        filing_data["Form D Filing"] = row.get("Form D Filing", "")
                        
                        detailed_data.append(filing_data)

                progress_bar.empty()
                status_text.empty()

                if detailed_data:
                    # Store timestamp
                    st.session_state["last_search_time"] = datetime.now()
                    
                    year_display = ", ".join(selected_years) if selected_years else "All years"
                    stages_display = ", ".join([s if s != "N/A" else "N/A (Unknown)" for s in allowed_stages]) if allowed_stages else "All stages"
                    st.success(f"✅ Found {len(detailed_data):,} matching funds ({industry_subtype if industry_subtype != 'Any' else 'Any subtype'}, Stages: {stages_display}, Years: {year_display}, Size: ${min_size}M-${max_size}M)")
                    
                    # Show last updated timestamp
                    if st.session_state["last_search_time"]:
                        # Convert to Mountain Time
                        utc_time = st.session_state["last_search_time"]
                        mountain = pytz.timezone('America/Denver')
                        utc_time = pytz.utc.localize(utc_time) if utc_time.tzinfo is None else utc_time
                        mountain_time = utc_time.astimezone(mountain)
                        timestamp = mountain_time.strftime("%m/%d/%Y at %I:%M %p MST")
                        st.caption(f"Last Updated: {timestamp}")
                    
                    # Create main view DataFrame
                    main_df = create_main_view_df(detailed_data)
                    
                    # Add SEC Form D links to Fund Name column
                    def make_clickable(row_data):
                        name = row_data['Fund Name']
                        cik = row_data.get('CIK', '')
                        acc = row_data.get('Accession Number', '').replace("-", "")
                        primary_doc = row_data.get('Primary Document', f"{acc}.txt")
                        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{primary_doc}"
                        return f'<a href="{url}" target="_blank">{name}</a>'
                    
                    # Create full dataframe for linking
                    full_df_with_tech = pd.DataFrame(detailed_data)
                    main_df_display = main_df.copy()
                    main_df_display['Fund Name'] = full_df_with_tech.apply(make_clickable, axis=1)
                    
                    # Display with HTML rendering for clickable links
                    st.write(main_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    # Store for CSV download
                    full_df = pd.DataFrame(detailed_data)
                    st.session_state["detailed_results"] = full_df

                    # CSV Download button at top
                    csv_data = convert_df_to_csv(full_df)
                    filename = f"formd_results_{start_date}_{end_date}.csv"
                    st.download_button(
                        label="📥 Download Complete Results as CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                    # Expandable details
                    st.write("---")
                    st.subheader("📂 Detailed Fund Information")
                    
                    for idx, row_data in enumerate(detailed_data):
                        with st.expander(f"🔍 {row_data['Fund Name']} - {row_data['Fund Stage']} ({row_data['Investment Type']})"):
                            st.markdown(create_expandable_section(row_data))
                    
                    # CSV Download button at bottom too
                    st.download_button(
                        label="📥 Download Complete Results as CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        key="download_bottom"
                    )
                else:
                    st.warning(f"No filings matched the selected filters out of {total_fetched:,} filings searched.")
            else:
                st.write(f"No filings found for {start_date} to {end_date}, or an error occurred.")

if __name__ == "__main__":
    main()
