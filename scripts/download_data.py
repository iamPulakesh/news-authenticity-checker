import os
import sys
import csv
import json
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NewsChecker/1.0)"
}

def download_snopes_rss() -> int:
    """Scrape Snopes RSS feed for recent fact-checks."""
    logger.info("Downloading Snopes RSS fact-checks...")

    import xml.etree.ElementTree as ET

    output_path  = RAW_DIR / "snopes_factchecks.csv"
    rows_written = 0

    # Snopes has category-specific RSS feeds
    rss_feeds = [
        ("https://www.snopes.com/fact-check/rating/false/feed/",       "false"),
        ("https://www.snopes.com/fact-check/rating/true/feed/",        "true"),
        ("https://www.snopes.com/fact-check/rating/mixture/feed/",     "misleading"),
        ("https://www.snopes.com/fact-check/rating/unproven/feed/",    "unverified"),
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "verdict", "source", "url", "description"])

        for feed_url, verdict in rss_feeds:
            try:
                resp = requests.get(feed_url, headers=HEADERS, timeout=20)
                resp.raise_for_status()

                root = ET.fromstring(resp.content)
                ns   = {"content": "http://purl.org/rss/1.0/modules/content/"}

                items = root.findall(".//item")
                for item in items:
                    title = item.findtext("title", "").strip()
                    link  = item.findtext("link", "").strip()
                    desc  = item.findtext("description", "").strip()

                    # Clean HTML tags from description
                    import re
                    desc = re.sub(r"<[^>]+>", "", desc).strip()

                    if title and len(title) > 15:
                        writer.writerow([title, verdict, "Snopes", link, desc[:500]])
                        rows_written += 1

                logger.info(f"  Fetched {len(items)} '{verdict}' checks from Snopes RSS")
                time.sleep(1)

            except Exception as e:
                logger.warning(f"  Snopes RSS feed failed ({feed_url}): {e}")

    logger.info(f"  Saved {rows_written} rows → {output_path}")
    return rows_written


def create_seed_dataset() -> int:
    """
    Create a hand-curated seed dataset of well-known fact-check examples.
    This guarantees the RAG has some data even if all downloads fail.
    Covers common misinformation patterns the agent will encounter.
    """
    logger.info("Creating curated seed fact-check dataset...")

    output_path  = RAW_DIR / "seed_factchecks.csv"

    seed_data = [
        {
            "claim": "A photograph shows Hillary Clinton shaking hands with Osama bin Laden",
            "verdict": "false",
            "explanation": "This image is digitally manipulated. No such meeting occurred. The original photo showed Clinton meeting with Pakistani officials.",
            "source": "Snopes",
            "category": "image_manipulation"
        },
        {
            "claim": "The COVID-19 vaccine contains microchips to track people",
            "verdict": "false",
            "explanation": "COVID-19 vaccines contain mRNA or viral vector components. No microchips, tracking devices, or any electronic components are present. This has been verified by multiple independent laboratory analyses.",
            "source": "WHO / Reuters Fact Check",
            "category": "health_misinformation"
        },
        {
            "claim": "5G towers cause COVID-19 or spread the coronavirus",
            "verdict": "false",
            "explanation": "COVID-19 is caused by the SARS-CoV-2 virus, which spreads through respiratory droplets. Radio waves from 5G towers cannot carry or transmit viruses. Countries without 5G also had COVID-19 outbreaks.",
            "source": "WHO / Full Fact",
            "category": "health_misinformation"
        },
        {
            "claim": "The Great Wall of China is visible from space with the naked eye",
            "verdict": "false",
            "explanation": "Chinese astronaut Yang Liwei confirmed he could not see the Great Wall from space. NASA has also clarified this is a myth — the wall is too narrow to be seen from low Earth orbit without aid.",
            "source": "NASA / National Geographic",
            "category": "historical_myth"
        },
        {
            "claim": "Drinking bleach or disinfectant can cure COVID-19",
            "verdict": "false",
            "explanation": "Ingesting bleach or disinfectants is extremely dangerous and potentially fatal. No medical authority endorses this. The FDA and CDC have issued strong warnings against this dangerous misinformation.",
            "source": "FDA / CDC",
            "category": "health_misinformation"
        },

        {
            "claim": "Sea levels have not risen in 50 years according to new data",
            "verdict": "misleading",
            "explanation": "Sea level rise data from NOAA and NASA shows consistent rise of 3-4mm per year since 1993. Cherry-picked local tide gauge data from specific locations may show variation but global trend is clear.",
            "source": "NOAA / NASA",
            "category": "climate_misinformation"
        },
        {
            "claim": "Immigrants commit more crimes than native-born citizens",
            "verdict": "false",
            "explanation": "Multiple peer-reviewed studies show immigrants, including undocumented immigrants, commit crimes at lower rates than native-born citizens. National Academy of Sciences 2020 report confirmed this finding.",
            "source": "National Academy of Sciences",
            "category": "political_misinformation"
        },
        {
            "claim": "Einstein failed math in school",
            "verdict": "false",
            "explanation": "Einstein excelled at mathematics from an early age. By 12 he had mastered calculus. This myth likely arose from a misunderstanding of Swiss grading systems where 6 is the top grade.",
            "source": "Historical records / Einstein Archives",
            "category": "historical_myth"
        },

        {
            "claim": "Humans share approximately 98% of their DNA with chimpanzees",
            "verdict": "true",
            "explanation": "Genetic studies confirm humans and chimpanzees share approximately 98.7% of their DNA sequence. This is one of the strongest pieces of evidence for our shared evolutionary history.",
            "source": "Nature / NCBI",
            "category": "science"
        },
        {
            "claim": "The Amazon rainforest produces 20% of the world's oxygen",
            "verdict": "misleading",
            "explanation": "The Amazon produces roughly 20% of global oxygen through photosynthesis, but it also consumes a similar amount through respiration of plants and organisms. Net oxygen contribution is near zero. The Amazon is critical for carbon sequestration, not oxygen supply.",
            "source": "Science Magazine / NASA",
            "category": "environment"
        },

        {
            "claim": "Vaccines cause autism",
            "verdict": "false",
            "explanation": "The original 1998 Wakefield study claiming this link was fraudulent and retracted. Wakefield lost his medical license. Over 30 large-scale studies involving millions of children have found no link between vaccines and autism.",
            "source": "Lancet retraction / CDC / WHO",
            "category": "health_misinformation"
        },
        {
            "claim": "Eating carrots improves your eyesight beyond normal vision",
            "verdict": "misleading",
            "explanation": "Carrots contain beta-carotene which the body converts to Vitamin A, essential for normal vision. Deficiency causes night blindness. However, eating extra carrots beyond nutritional needs does not improve vision in people with adequate Vitamin A.",
            "source": "American Academy of Ophthalmology",
            "category": "health"
        },

        {
            "claim": "Facebook listens to your conversations through your phone microphone to serve targeted ads",
            "verdict": "unverified",
            "explanation": "Meta/Facebook denies this practice. Security researchers have found no concrete evidence of constant audio recording. However, the company's extensive behavioral and demographic targeting creates an illusion of eavesdropping. The claim remains unproven but cannot be fully ruled out.",
            "source": "EFF / Tech researchers",
            "category": "technology"
        },
        {
            "claim": "ChatGPT and AI will eliminate 300 million jobs by 2030",
            "verdict": "misleading",
            "explanation": "A Goldman Sachs report estimated AI could affect 300 million full-time jobs but also stated it would create new roles. The report discussed automation potential, not guaranteed job elimination. Economic disruption is likely but the net effect is debated.",
            "source": "Goldman Sachs Report 2023",
            "category": "technology"
        },

        {
            "claim": "The 2020 US Presidential election was stolen through widespread fraud",
            "verdict": "false",
            "explanation": "Over 60 courts including judges appointed by both parties found no evidence of widespread fraud. The Department of Justice, CISA, and state election officials confirmed the election was secure. Multiple audits confirmed Biden's victory.",
            "source": "US Courts / DOJ / CISA",
            "category": "political_misinformation"
        },
        {
            "claim": "Climate change is a hoax invented by scientists for research grants",
            "verdict": "false",
            "explanation": "97% of actively publishing climate scientists agree human-caused climate change is real. Evidence comes from multiple independent lines: temperature records, ice cores, sea level rise, ocean acidification. The scientific consensus is overwhelming.",
            "source": "NASA / IPCC / Multiple peer-reviewed journals",
            "category": "climate_misinformation"
        },
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["claim", "verdict", "explanation", "source", "category"])
        writer.writeheader()
        writer.writerows(seed_data)

    logger.info(f"  Saved {len(seed_data)} seed fact-checks -> {output_path}")
    return len(seed_data)


def create_indian_seed_dataset() -> int:
    """
    Curated Indian fact-check dataset covering:
      - Healthcare & Ayurveda misinformation
      - Indian politics & elections
      - AI & technology in India
      - Economy, GDP & RBI policies
      - Social media viral misinformation
    """
    logger.info("Creating curated Indian fact-check dataset...")

    output_path = RAW_DIR / "indian_factchecks.csv"

    seed_data = [
        {
            "claim": "Drinking cow urine (gomutra) cures COVID-19 and other viral diseases",
            "verdict": "false",
            "explanation": "No scientific evidence supports cow urine as a treatment for COVID-19 or any viral disease. ICMR and WHO have both stated there is no clinical data supporting this claim. Cow urine can contain harmful bacteria.",
            "source": "ICMR / WHO / AltNews",
            "category": "health_misinformation"
        },
        {
            "claim": "AIIMS Delhi confirmed that Coronil cures COVID-19",
            "verdict": "false",
            "explanation": "AIIMS Delhi denied any involvement in trials for Patanjali's Coronil. The WHO also clarified it never certified Coronil as a COVID-19 cure. Patanjali later positioned it as an immunity booster rather than a cure.",
            "source": "AIIMS Delhi / BOOM Live",
            "category": "health_misinformation"
        },
        {
            "claim": "India's COVID-19 death toll was significantly undercounted by a factor of 6-7x",
            "verdict": "misleading",
            "explanation": "Multiple independent studies (Lancet, WHO excess mortality estimates) suggest India's actual COVID death toll was significantly higher than officially reported, potentially 3-6x. However, exact undercount varies by methodology and region. The government disputed these estimates.",
            "source": "The Lancet / WHO / BOOM Live",
            "category": "health"
        },
        {
            "claim": "5G towers installed across India are spreading coronavirus",
            "verdict": "false",
            "explanation": "5G radio waves cannot carry or transmit viruses. COVID-19 is transmitted via respiratory droplets. India had COVID cases well before any 5G deployment. The Department of Telecom dismissed this claim as baseless conspiracy theory.",
            "source": "DoT India / PIB Fact Check",
            "category": "health_misinformation"
        },
        {
            "claim": "Ayurvedic kadha with tulsi, ginger and turmeric is a proven cure for COVID-19",
            "verdict": "misleading",
            "explanation": "These ingredients may support general immunity and have traditional medicinal uses in Ayurveda. However, no clinical trial has proven they cure COVID-19 specifically. AYUSH ministry recommended them as supportive measures, not cures.",
            "source": "AYUSH Ministry / ICMR",
            "category": "health"
        },
        {
            "claim": "India's Chandrayaan-3 successfully landed on the Moon's south pole, a world first",
            "verdict": "true",
            "explanation": "On August 23, 2023, ISRO's Chandrayaan-3 Vikram lander successfully touched down near the Moon's south polar region, making India the first country to land near the lunar south pole and the fourth country to achieve a soft landing on the Moon.",
            "source": "ISRO / NASA / BBC",
            "category": "science_technology"
        },
        {
            "claim": "India has completely eradicated polio and was certified polio-free by WHO",
            "verdict": "true",
            "explanation": "India was officially declared polio-free by the WHO South-East Asia Region in March 2014, after zero cases for three consecutive years. The last polio case was recorded in January 2011. This was achieved through massive immunization campaigns.",
            "source": "WHO / UNICEF / PIB",
            "category": "health"
        },

        {
            "claim": "India's Parliament passed a law making Aadhaar mandatory for all banking transactions",
            "verdict": "misleading",
            "explanation": "While the Aadhaar Act allows banks to use Aadhaar for KYC, the Supreme Court of India ruled in 2018 (Puttaswamy judgment) that Aadhaar cannot be made mandatory for bank accounts. Banks must accept alternative ID documents.",
            "source": "Supreme Court of India / RBI circulars",
            "category": "politics"
        },
        {
            "claim": "The Indian government banned all cryptocurrency trading and usage in India",
            "verdict": "false",
            "explanation": "India has not banned cryptocurrency. The Supreme Court struck down the RBI's banking ban on crypto in 2020. The government introduced a 30% tax on crypto gains and 1% TDS in Budget 2022, but crypto trading remains legal.",
            "source": "Supreme Court / RBI / Finance Ministry",
            "category": "economy"
        },
        {
            "claim": "EVM machines in Indian elections were hacked to change election results",
            "verdict": "unverified",
            "explanation": "The Election Commission of India maintains EVMs are standalone machines with no internet connectivity, making remote hacking impossible. Multiple EVM hackathon challenges have been issued. Opposition parties have raised concerns but no concrete proof of tampering has been established in any court.",
            "source": "Election Commission of India / Supreme Court",
            "category": "politics"
        },
        {
            "claim": "PM-KISAN scheme deposits Rs 6000 per year directly to all farmers' bank accounts",
            "verdict": "misleading",
            "explanation": "PM-KISAN provides Rs 6000/year in 3 installments to eligible farmer families, not all farmers. Exclusions include institutional landholders, income taxpayers, and government employees. As of 2024, over 11 crore farmers are registered but many complaints of non-receipt exist.",
            "source": "PIB / PM-KISAN Portal / PRS Legislative Research",
            "category": "politics"
        },
        {
            "claim": "India's Uniform Civil Code has been passed and applied nationwide",
            "verdict": "false",
            "explanation": "As of early 2025, no nationwide Uniform Civil Code has been enacted. Uttarakhand passed a state-level UCC in 2024. The UCC remains a policy proposal under Article 44 of the Constitution (Directive Principles). No central legislation has been tabled in Parliament.",
            "source": "PRS Legislative Research / The Hindu",
            "category": "politics"
        },
        {
            "claim": "Voter turnout data from Election Commission shows anomalies proving rigged elections",
            "verdict": "misleading",
            "explanation": "Discrepancies between initial turnout percentages announced on election day and final certified figures are normal. Final counts include postal ballots and corrections from presiding officers. The Election Commission has explained the reconciliation process in detail.",
            "source": "Election Commission of India / FactChecker.in",
            "category": "politics"
        },

        {
            "claim": "India has more AI startups than China and is the world's second largest AI hub",
            "verdict": "misleading",
            "explanation": "While India has a rapidly growing AI ecosystem and ranks among the top 5 globally for AI startups, China significantly leads India in AI investment, patents, and number of AI companies. Exact rankings vary by methodology — Stanford AI Index, OECD, and Nasscom reports provide different figures.",
            "source": "Stanford AI Index 2024 / Nasscom / OECD",
            "category": "technology"
        },
        {
            "claim": "ChatGPT and AI tools will eliminate 50% of IT jobs in India by 2025",
            "verdict": "misleading",
            "explanation": "While AI will transform the IT industry, predictions of 50% job elimination by 2025 are greatly exaggerated. Nasscom estimates AI will change the nature of 30-40% of IT roles but also create new ones. McKinsey and WEF report net positive job creation from AI in emerging economies like India.",
            "source": "Nasscom / McKinsey / World Economic Forum",
            "category": "technology"
        },
        {
            "claim": "India's UPI processed more digital transactions than all credit cards globally combined",
            "verdict": "misleading",
            "explanation": "UPI processes a massive volume of transactions — over 10 billion monthly by 2024. Comparing transaction count (dominated by small-value UPI payments) to credit card transaction value is misleading. By value, global credit card transactions still far exceed UPI. By volume, UPI leads.",
            "source": "NPCI / RBI / Visa Reports",
            "category": "technology"
        },
        {
            "claim": "India's digital public infrastructure (Aadhaar, UPI, DigiLocker) is being adopted by 40+ countries",
            "verdict": "misleading",
            "explanation": "India has signed MoUs with several countries for UPI-like systems and shared its DPI frameworks. However, 'adoption' is overstated — most countries are studying the framework, not implementing identical systems. UPI has launched interoperability with Singapore, UAE, and a few others.",
            "source": "NPCI International / MEA / MOSIP",
            "category": "technology"
        },
        {
            "claim": "Reliance Jio's AI model is more powerful than OpenAI's GPT-4",
            "verdict": "unverified",
            "explanation": "Reliance has announced AI initiatives and partnerships but has not publicly released independent benchmarks comparing their models to GPT-4. Without peer-reviewed evaluations or standardized benchmark results, this claim remains unverified.",
            "source": "Tech industry reports",
            "category": "technology"
        },
        {
            "claim": "Indian IT companies are replacing 80% of their workforce with AI automation",
            "verdict": "false",
            "explanation": "Major Indian IT companies (TCS, Infosys, Wipro) are investing in AI but have clarified they are reskilling workers, not mass-replacing them. Industry-wide layoffs are driven by multiple factors including business restructuring, not solely AI. Nasscom data shows overall IT employment continues to grow.",
            "source": "Nasscom / Company annual reports / Economic Times",
            "category": "technology"
        },

        {
            "claim": "India's GDP growth rate of 8.2% in FY24 makes it the fastest growing major economy",
            "verdict": "true",
            "explanation": "India recorded 8.2% real GDP growth in FY2023-24 according to NSO estimates, making it the fastest growing major economy ahead of China (5.2%). The IMF and World Bank confirmed India's leading growth position among G20 nations.",
            "source": "NSO / IMF / World Bank",
            "category": "economy"
        },
        {
            "claim": "GST has completely eliminated tax evasion and black money in India",
            "verdict": "false",
            "explanation": "While GST improved tax compliance and formalized parts of the economy, tax evasion has not been eliminated. GST fraud cases worth thousands of crores are regularly detected. The system has improved transparency but challenges with fake invoicing, under-reporting, and the informal sector persist.",
            "source": "GST Council / CAG Reports / Economic Survey",
            "category": "economy"
        },
        {
            "claim": "India became the world's fifth largest economy surpassing the United Kingdom",
            "verdict": "true",
            "explanation": "According to IMF data, India overtook the UK to become the fifth largest economy by nominal GDP in 2022. By 2024, India maintained this position and is projected to become the third largest economy by 2027-28.",
            "source": "IMF World Economic Outlook / World Bank",
            "category": "economy"
        },
        {
            "claim": "Demonetisation in 2016 permanently eliminated black money from India's economy",
            "verdict": "false",
            "explanation": "The RBI reported that 99.3% of demonetised currency was returned to banks, suggesting most black money held in cash was successfully converted. The stated objective of eliminating black money was not fully achieved. Black money exists in multiple forms beyond cash including real estate, gold, and foreign accounts.",
            "source": "RBI Annual Report / CAG / Economic Survey",
            "category": "economy"
        },
        {
            "claim": "India's unemployment rate is at an all-time high of 45%",
            "verdict": "false",
            "explanation": "India's unemployment rate according to CMIE (Centre for Monitoring Indian Economy) fluctuates between 7-9%. The PLFS (Periodic Labour Force Survey) by NSO shows rates around 3-4% using usual status definition. A 45% figure has no basis in any official or credible independent survey.",
            "source": "CMIE / NSO PLFS / ILO",
            "category": "economy"
        },
        {
            "claim": "The Indian rupee has lost 50% of its value against the dollar under the current government",
            "verdict": "misleading",
            "explanation": "The rupee depreciated from approximately Rs 60/USD in 2014 to around Rs 83-84/USD by 2024 — roughly a 40% nominal depreciation over a decade. However, this must be contextualized with inflation differentials between India and the US, which make real effective exchange rate depreciation much smaller. Currency depreciation also occurred under previous governments.",
            "source": "RBI / IMF / Bloomberg",
            "category": "economy"
        },

        {
            "claim": "A viral WhatsApp forward claims drinking warm water with lemon kills all viruses",
            "verdict": "false",
            "explanation": "Warm lemon water has nutritional benefits but cannot kill viruses. Multiple doctors and ICMR have confirmed that no beverage can kill pathogens within the body. This claim has been repeatedly debunked by Indian fact-checkers including AltNews and BOOM.",
            "source": "ICMR / AltNews / BOOM Live",
            "category": "health_misinformation"
        },
        {
            "claim": "NASA declared that India was the most polluted country and Delhi the most polluted city in the world",
            "verdict": "misleading",
            "explanation": "IQAir (not NASA) publishes air quality reports. Delhi frequently ranks among the most polluted cities, but 'most polluted country' is an oversimplification. India has some of the world's most polluted cities but pollution varies vastly by region. WHO/IQAir data ranks by city, not country.",
            "source": "IQAir / WHO / AltNews",
            "category": "environment"
        },
        {
            "claim": "A photo shows Indian soldiers planting the national flag on Pakistani soil during the Balakot airstrike",
            "verdict": "false",
            "explanation": "Multiple viral photos claiming to show the Balakot airstrike aftermath were debunked as old images or images from military exercises. The Indian Air Force confirmed the airstrike but did not release ground photographs. Fact-checkers traced the viral images to unrelated events.",
            "source": "AltNews / BOOM Live / PIB Fact Check",
            "category": "image_manipulation"
        },
        {
            "claim": "RBI announced a new Rs 1000 note with a GPS tracking chip inside",
            "verdict": "false",
            "explanation": "The RBI never announced GPS-embedded currency notes. This viral claim resurfaces periodically. Current technology does not allow GPS chips small enough to be embedded in paper currency. The RBI has publicly denied this claim multiple times.",
            "source": "RBI / PIB Fact Check",
            "category": "technology"
        },
        {
            "claim": "India's population has officially overtaken China according to the UN, making India the most populous country",
            "verdict": "true",
            "explanation": "The UN Population Fund (UNFPA) estimated in April 2023 that India's population reached 1.4286 billion, surpassing China's 1.4257 billion, making India the world's most populous country. This was subsequently confirmed by multiple demographic sources.",
            "source": "UNFPA / UN Population Division / World Bank",
            "category": "demographics"
        },
        {
            "claim": "A viral video shows Muslims in India being forced to chant Hindu slogans",
            "verdict": "misleading",
            "explanation": "While some verified incidents of communal coercion have been documented and prosecuted, many viral videos lack context, are old clips recirculated, or are from different countries entirely. Each incident must be verified individually rather than treated as representative of a widespread pattern.",
            "source": "AltNews / BOOM Live / India Today Fact Check",
            "category": "communal_misinformation"
        },
        {
            "claim": "India's ISRO budget is smaller than NASA's parking lot budget",
            "verdict": "misleading",
            "explanation": "This claim became popular after Mangalyaan's low-cost Mars mission. While ISRO's entire annual budget (around $1.5-2 billion) is a small fraction of NASA's ($25+ billion), the 'parking lot' comparison is a made-up figure with no basis. NASA does not have a publicized parking lot budget. ISRO achieves remarkable cost efficiency.",
            "source": "ISRO / NASA / PRS Legislative Research",
            "category": "science_technology"
        },
        {
            "claim": "Adani Group's stock crash in 2023 was caused by a foreign conspiracy against India",
            "verdict": "misleading",
            "explanation": "The Adani stock crash was triggered by a research report by US short-seller Hindenburg Research alleging accounting fraud and stock manipulation. While some argued it was a targeted attack, the report cited specific financial data. SEBI investigated the claims. The characterization as a 'foreign conspiracy against India' is a political framing, not a factual assessment.",
            "source": "SEBI / Hindenburg Research / Supreme Court Expert Committee",
            "category": "economy"
        },
        {
            "claim": "India's new education policy NEP 2020 makes Sanskrit compulsory in all schools",
            "verdict": "false",
            "explanation": "NEP 2020 does not make Sanskrit or any specific language compulsory. It recommends teaching in the mother tongue or regional language, with a three-language formula that gives states flexibility. Sanskrit is mentioned as an option alongside other classical and foreign languages.",
            "source": "NEP 2020 document / Ministry of Education / PRS",
            "category": "politics"
        },
        {
            "claim": "WhatsApp forwards claiming RBI will charge fees for UPI transactions starting 2024",
            "verdict": "false",
            "explanation": "The RBI and NPCI confirmed that UPI transactions remain free for users as of 2024. While interchange fees exist between banks, these are not charged to end users. Government has repeatedly stated UPI will remain zero-cost for consumers. Viral WhatsApp messages about UPI charges are false.",
            "source": "RBI / NPCI / Finance Ministry",
            "category": "technology"
        },
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["claim", "verdict", "explanation", "source", "category"])
        writer.writeheader()
        writer.writerows(seed_data)

    logger.info(f"  Saved {len(seed_data)} Indian fact-checks -> {output_path}")
    return len(seed_data)


def download_boom_rss() -> int:
    """Scrape BOOM Live RSS feed for Indian fact-checks."""
    logger.info("Downloading BOOM Live (Indian fact-checker) RSS...")

    import xml.etree.ElementTree as ET
    import re

    output_path  = RAW_DIR / "boom_factchecks.csv"
    rows_written = 0

    # BOOM Live fact-check RSS feeds
    rss_feeds = [
        ("https://www.boomlive.in/fact-check/feed", "mixed"),
        ("https://www.boomlive.in/fake-news/feed",  "false"),
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "verdict", "source", "url", "description"])

        for feed_url, default_verdict in rss_feeds:
            try:
                resp = requests.get(feed_url, headers=HEADERS, timeout=20)
                resp.raise_for_status()

                root = ET.fromstring(resp.content)
                items = root.findall(".//item")

                for item in items:
                    title = item.findtext("title", "").strip()
                    link  = item.findtext("link", "").strip()
                    desc  = item.findtext("description", "").strip()

                    # Clean HTML tags from description
                    desc = re.sub(r"<[^>]+>", "", desc).strip()

                    # Detect verdict from title keywords
                    title_lower = title.lower()
                    if "false" in title_lower or "fake" in title_lower or "no," in title_lower:
                        verdict = "false"
                    elif "true" in title_lower or "yes," in title_lower:
                        verdict = "true"
                    elif "misleading" in title_lower or "missing context" in title_lower:
                        verdict = "misleading"
                    else:
                        verdict = default_verdict

                    if title and len(title) > 15:
                        writer.writerow([title, verdict, "BOOM Live", link, desc[:500]])
                        rows_written += 1

                logger.info(f"  Fetched {len(items)} checks from BOOM Live ({feed_url.split('/')[-2]})")
                time.sleep(1)

            except Exception as e:
                logger.warning(f"  BOOM RSS feed failed ({feed_url}): {e}")

    logger.info(f"  Saved {rows_written} rows -> {output_path}")
    return rows_written


def download_liar_dataset() -> int:
    """
    Download the LIAR dataset — 12,836 labeled political claims from PolitiFact.
    Labels: pants-fire, false, barely-true, half-true, mostly-true, true
    Source: Wang 2017, "Liar, Liar Pants on Fire"
    """
    logger.info("Downloading LIAR dataset (political claims)...")

    output_path  = RAW_DIR / "liar_factchecks.csv"
    rows_written = 0

    # LIAR dataset TSV files hosted on GitHub mirrors
    liar_urls = [
        "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
        "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv",
        "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
    ]

    # Map LIAR 6-label to our 4-label scheme
    label_map = {
        "pants-fire":   "false",
        "false":        "false",
        "barely-true":  "misleading",
        "half-true":    "misleading",
        "mostly-true":  "true",
        "true":         "true",
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "verdict", "source", "category"])

        for url in liar_urls:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()

                for line in resp.text.strip().splitlines():
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue

                    # LIAR TSV: col0=id, col1=label, col2=statement, ...
                    raw_label  = parts[1].strip().lower()
                    statement  = parts[2].strip()
                    speaker    = parts[4].strip() if len(parts) > 4 else ""
                    context    = parts[-1].strip() if len(parts) > 5 else ""

                    if len(statement) < 20:
                        continue

                    verdict = label_map.get(raw_label, "unverified")
                    source_info = f"PolitiFact ({speaker})" if speaker else "PolitiFact"

                    writer.writerow([statement, verdict, source_info, "politics"])
                    rows_written += 1

                split_name = url.split("/")[-1]
                logger.info(f"  Parsed {split_name}")
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"  LIAR download failed ({url}): {e}")

    logger.info(f"  Saved {rows_written} rows -> {output_path}")
    return rows_written


def download_google_factcheck() -> int:
    """
    Fetch fact-checks from Google Fact Check Tools API (ClaimSearch).
    Uses API key from config for higher quota (10,000 req/day free).
    Queries 50+ topics across all categories with pagination.
    """
    logger.info("Fetching from Google Fact Check API...")

    from app.config import settings

    output_path  = RAW_DIR / "google_factchecks.csv"
    rows_written = 0
    seen_claims  = set()

    api_key = settings.GOOGLE_FACTCHECK_API_KEY

    # 50+ diverse queries across all underrepresented categories
    search_queries = [
        # Healthcare
        "COVID vaccine side effects", "ivermectin COVID cure", "vaccine causes autism",
        "cancer cure breakthrough", "diabetes cure natural", "WHO health warning",
        "monkeypox virus", "bird flu pandemic", "antibiotics resistance",
        "heart attack warning signs viral", "drinking water cure disease",
        # AI & Technology
        "AI jobs replacement", "ChatGPT dangers", "deepfake video",
        "AI sentient", "social media privacy", "5G health risks",
        "robot replace workers", "AI bias discrimination", "facial recognition ban",
        "cryptocurrency scam", "bitcoin crash", "NFT fraud",
        # Economy & Finance
        "economic recession 2025", "inflation rate misleading", "stock market crash",
        "bank collapse", "unemployment rate", "GDP growth false",
        "housing market bubble", "minimum wage", "tax policy",
        "trade deficit", "national debt", "cost of living crisis",
        # Science & Environment
        "climate change hoax", "global warming fake", "sea level rise",
        "deforestation Amazon", "electric vehicle battery", "nuclear energy safe",
        "GMO food dangerous", "pesticide cancer", "renewable energy cost",
        "plastic pollution ocean", "species extinction rate",
        # Social/Viral Misinformation
        "viral video fake", "photo manipulated", "WhatsApp forward hoax",
        "celebrity death hoax", "natural disaster fake", "election fraud",
        "immigration crime statistics", "flat earth proof", "moon landing fake",
        # India-specific
        "India COVID", "India economy growth", "India election",
        "India vaccine", "India technology", "Modi government",
        "Aadhaar privacy", "UPI digital payment", "India pollution",
        "India education policy", "India healthcare scheme", "India farmer protest",
    ]

    # Map Google ClaimReview textualRating to our labels
    def map_rating(rating_text: str) -> str:
        rating = rating_text.lower()
        if any(w in rating for w in ["false", "pants on fire", "incorrect", "wrong", "no", "fake", "fabricat"]):
            return "false"
        if any(w in rating for w in ["true", "correct", "accurate", "yes"]):
            return "true"
        if any(w in rating for w in ["misleading", "partly", "half", "mixture", "distort", "exaggerat", "missing context"]):
            return "misleading"
        return "unverified"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "verdict", "source", "url", "explanation"])

        for query in search_queries:
            next_page_token = None
            pages_fetched   = 0
            max_pages       = 5  # up to 5 pages per query

            while pages_fetched < max_pages:
                try:
                    api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
                    params = {
                        "query": query,
                        "pageSize": 10,
                        "languageCode": "en",
                    }
                    if api_key:
                        params["key"] = api_key
                    if next_page_token:
                        params["pageToken"] = next_page_token

                    resp = requests.get(api_url, params=params, headers=HEADERS, timeout=15)

                    if resp.status_code != 200:
                        break

                    data = resp.json()
                    claims = data.get("claims", [])

                    if not claims:
                        break

                    batch_count = 0
                    for item in claims:
                        claim_text = item.get("text", "").strip()
                        if not claim_text or len(claim_text) < 15:
                            continue

                        # Dedup
                        claim_key = claim_text[:80].lower()
                        if claim_key in seen_claims:
                            continue
                        seen_claims.add(claim_key)

                        reviews = item.get("claimReview", [])
                        if not reviews:
                            continue

                        review    = reviews[0]
                        publisher = review.get("publisher", {}).get("name", "Unknown")
                        rating    = review.get("textualRating", "Unknown")
                        url       = review.get("url", "")
                        title     = review.get("title", "")

                        verdict = map_rating(rating)

                        writer.writerow([
                            claim_text,
                            verdict,
                            publisher,
                            url,
                            f"Rating: {rating}. {title}"[:500],
                        ])
                        rows_written += 1
                        batch_count += 1

                    next_page_token = data.get("nextPageToken")
                    pages_fetched += 1

                    if not next_page_token:
                        break

                    time.sleep(0.2)

                except Exception as e:
                    logger.warning(f"  Google FC query '{query}' page {pages_fetched} failed: {e}")
                    break

            if pages_fetched > 0:
                logger.info(f"  '{query}' -> {pages_fetched} pages fetched")

            time.sleep(0.15)

    logger.info(f"  Saved {rows_written} rows -> {output_path}")
    return rows_written




def _download_rss_generic(
    feed_urls: list[tuple[str, str]],
    output_path: Path,
    source_name: str,
) -> int:
    """
    Generic RSS feed downloader.
    feed_urls: list of (url, default_verdict) tuples.
    """
    import xml.etree.ElementTree as ET
    import re

    rows_written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["claim", "verdict", "source", "url", "description"])

        for feed_url, default_verdict in feed_urls:
            try:
                resp = requests.get(feed_url, headers=HEADERS, timeout=20)
                resp.raise_for_status()

                root = ET.fromstring(resp.content)
                items = root.findall(".//item")

                for item in items:
                    title = item.findtext("title", "").strip()
                    link  = item.findtext("link", "").strip()
                    desc  = item.findtext("description", "").strip()

                    # Clean HTML
                    desc = re.sub(r"<[^>]+>", "", desc).strip()

                    # Try to detect verdict from title
                    title_lower = title.lower()
                    if any(w in title_lower for w in ["false", "fake", "no,", "wrong", "fabricat", "debunk"]):
                        verdict = "false"
                    elif any(w in title_lower for w in ["true", "correct", "yes,", "confirmed"]):
                        verdict = "true"
                    elif any(w in title_lower for w in ["misleading", "missing context", "partly", "half", "exaggerat"]):
                        verdict = "misleading"
                    else:
                        verdict = default_verdict

                    if title and len(title) > 15:
                        writer.writerow([title, verdict, source_name, link, desc[:500]])
                        rows_written += 1

                logger.info(f"  {source_name}: {len(items)} items from {feed_url.split('/')[2]}")
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"  {source_name} RSS failed ({feed_url}): {e}")

    logger.info(f"  Saved {rows_written} rows -> {output_path}")
    return rows_written


def download_fullfact_rss() -> int:
    """Download FullFact (UK) fact-checks — covers economy, health, immigration, law."""
    logger.info("Downloading FullFact RSS (health, economy, immigration)...")
    return _download_rss_generic(
        feed_urls=[
            ("https://fullfact.org/feed/", "misleading"),
        ],
        output_path=RAW_DIR / "fullfact_factchecks.csv",
        source_name="FullFact",
    )


def download_health_science_rss() -> int:
    """
    Download from Health Feedback, Science Feedback, Climate Feedback.
    These are specialized fact-checkers for science/health/climate claims.
    """
    logger.info("Downloading Health/Science/Climate Feedback RSS...")
    return _download_rss_generic(
        feed_urls=[
            ("https://healthfeedback.org/feed/", "misleading"),
            ("https://sciencefeedback.co/feed/", "misleading"),
            ("https://climatefeedback.org/feed/", "misleading"),
        ],
        output_path=RAW_DIR / "science_health_factchecks.csv",
        source_name="Science/Health Feedback",
    )


def download_factcheckorg_rss() -> int:
    """Download FactCheck.org RSS — covers health, science, economy, politics."""
    logger.info("Downloading FactCheck.org RSS (health, science, economy)...")
    return _download_rss_generic(
        feed_urls=[
            ("https://www.factcheck.org/feed/", "misleading"),
        ],
        output_path=RAW_DIR / "factcheckorg_factchecks.csv",
        source_name="FactCheck.org",
    )


def verify_downloads():
    """Check what data was downloaded and print a summary."""
    logger.info("Download Summary:")
    logger.info("─" * 50)

    total = 0
    for csv_file in sorted(RAW_DIR.glob("*.csv")):
        with open(csv_file, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1   # subtract header
        total += rows
        logger.info(f"  {csv_file.name:<35} {rows:>5} rows")

    logger.info(f"  {'TOTAL':<35} {total:>5} fact-checks")
    return total



if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Downloading Fact-Check Datasets")
    logger.info("=" * 55)

    total = 0

    # Always create seed data first (guaranteed to work offline)
    total += create_seed_dataset()
    total += create_indian_seed_dataset()


    rss_downloaders = [
        ("Snopes RSS",              download_snopes_rss),
        ("BOOM Live RSS",           download_boom_rss),
        ("FullFact RSS",            download_fullfact_rss),
        ("Health/Science Feedback", download_health_science_rss),
        ("FactCheck.org RSS",       download_factcheckorg_rss),
    ]

    for name, func in rss_downloaders:
        try:
            total += func()
        except Exception as e:
            logger.warning(f"{name} download skipped: {e}")


    try:
        total += download_liar_dataset()
    except Exception as e:
        logger.warning(f"LIAR dataset download skipped: {e}")


    try:
        total += download_google_factcheck()
    except Exception as e:
        logger.warning(f"Google Fact Check download skipped: {e}")

    final = verify_downloads()

    logger.info("\n" + "=" * 55)
    if final >= 10:
        logger.info(f"  Ready! {final} fact-checks downloaded.")
        logger.info("  Next step: python scripts/ingest_data.py")
    else:
        logger.info("  Very few records. Check your internet connection.")
    logger.info("=" * 55 + "\n")
