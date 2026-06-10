"""
utils/data.py
All static reference data: drug database, disease map, rationales, disease prevalence,
patient subgroups, regulatory lookup tables.
"""
from typing import Dict, List

# ---------------------------------------------------------------------------
# DRUG DATABASE
# ---------------------------------------------------------------------------
DRUG_DATA: Dict[str, Dict] = {
    "Metformin":          {"approved_indication":"Type 2 Diabetes","mechanism":"AMPK activator / mTOR inhibitor","clinical_phase":"Approved (1994)","protein_targets":["PRKAA1","MTOR","PPARG","TP53","SIRT1","IGF1R","PRKAA2","STK11"],"display_targets":["AMPK (PRKAA1)","mTORC1","PPARG","TP53","SIRT1","IGF1R"],"chembl_id":"CHEMBL1431"},
    "Berberine":          {"approved_indication":"Type 2 Diabetes / Hyperlipidemia","mechanism":"AMPK activator / PCSK9 inhibitor","clinical_phase":"Approved (China) / Phase II","protein_targets":["PRKAA1","PCSK9","LDLR","HMGCR","PPARG","NR1H4"],"display_targets":["PRKAA1","PCSK9","LDLR","HMGCR","PPARG"],"chembl_id":"CHEMBL47613"},
    "Aspirin":            {"approved_indication":"Pain / Inflammation / Antiplatelet","mechanism":"COX-1/2 inhibitor / Antiplatelet","clinical_phase":"Approved (1899)","protein_targets":["PTGS1","PTGS2","NFKB1","VEGFA","BCL2","PCNA","F2"],"display_targets":["PTGS1 (COX-1)","PTGS2 (COX-2)","NF-kB","VEGF","BCL2"],"chembl_id":"CHEMBL25"},
    "Atorvastatin":       {"approved_indication":"Hypercholesterolemia / CVD Prevention","mechanism":"HMG-CoA reductase inhibitor","clinical_phase":"Approved (1996)","protein_targets":["HMGCR","PCSK9","LDLR","ABCA1","CYP3A4","NPC1L1"],"display_targets":["HMGCR","PCSK9","LDLR","ABCA1","CYP3A4"],"chembl_id":"CHEMBL1487"},
    "Sildenafil":         {"approved_indication":"Erectile Dysfunction / PAH","mechanism":"PDE5 inhibitor","clinical_phase":"Approved (1998)","protein_targets":["PDE5A","PDE6A","PDE6B","NOS3","GUCY1A1"],"display_targets":["PDE5A","PDE6A","NOS3","GUCY1A1"],"chembl_id":"CHEMBL192"},
    "Imatinib":           {"approved_indication":"Chronic Myeloid Leukemia","mechanism":"BCR-ABL / KIT / PDGFR inhibitor","clinical_phase":"Approved (2001)","protein_targets":["ABL1","KIT","PDGFRA","PDGFRB","ABL2","CSF1R"],"display_targets":["BCR-ABL","KIT","PDGFRA","PDGFRB","ARG"],"chembl_id":"CHEMBL941"},
    "Rapamycin":          {"approved_indication":"Organ Transplant Rejection","mechanism":"mTORC1 inhibitor (via FKBP12)","clinical_phase":"Approved (1999)","protein_targets":["MTOR","FKBP1A","RPS6KB1","EIF4EBP1","AKT1","RICTOR"],"display_targets":["mTOR","FKBP12","S6K1","4EBP1","AKT1"],"chembl_id":"CHEMBL413"},
    "Doxorubicin":        {"approved_indication":"Breast / Lung / Ovarian Cancer","mechanism":"Topoisomerase II inhibitor / DNA intercalator","clinical_phase":"Approved (1974)","protein_targets":["TOP2A","TOP2B","TP53","ABCB1","BCL2","BAX"],"display_targets":["TOP2A","TOP2B","TP53","ABCB1","BCL2"],"chembl_id":"CHEMBL53463"},
    "Tamoxifen":          {"approved_indication":"Breast Cancer (ER+)","mechanism":"Selective estrogen receptor modulator (SERM)","clinical_phase":"Approved (1977)","protein_targets":["ESR1","ESR2","SHBG","CYP2D6","NCOA3","CCND1"],"display_targets":["ESR1","ESR2","SHBG","CYP2D6","NCOA3"],"chembl_id":"CHEMBL83"},
    "Gefitinib":          {"approved_indication":"Non-Small Cell Lung Cancer (EGFR+)","mechanism":"EGFR tyrosine kinase inhibitor","clinical_phase":"Approved (2003)","protein_targets":["EGFR","ERBB2","PIK3CA","AKT1","MTOR","MAPK1"],"display_targets":["EGFR","ERBB2","PIK3CA","AKT1","MTOR"],"chembl_id":"CHEMBL939"},
    "Dasatinib":          {"approved_indication":"CML / ALL (BCR-ABL+)","mechanism":"BCR-ABL / Src family kinase inhibitor","clinical_phase":"Approved (2006)","protein_targets":["ABL1","SRC","KIT","PDGFRA","EPHA2","YES1"],"display_targets":["ABL1","SRC","KIT","PDGFRA","EPHA2"],"chembl_id":"CHEMBL1421"},
    "Sorafenib":          {"approved_indication":"Hepatocellular / Renal Cell Carcinoma","mechanism":"Multi-kinase inhibitor (RAF/VEGFR/PDGFR)","clinical_phase":"Approved (2005)","protein_targets":["BRAF","RAF1","KDR","PDGFRB","KIT","FLT3"],"display_targets":["BRAF","RAF1","KDR (VEGFR2)","PDGFRB","KIT"],"chembl_id":"CHEMBL1336"},
    "Thalidomide":        {"approved_indication":"Multiple Myeloma / Leprosy","mechanism":"Cereblon (CRBN) modulator / anti-angiogenic","clinical_phase":"Approved (2006 for MM)","protein_targets":["CRBN","IKZF1","IKZF3","TNF","VEGFA","FGF2"],"display_targets":["CRBN","IKZF1","IKZF3","TNF","VEGFA"],"chembl_id":"CHEMBL468"},
    "Valproic Acid":      {"approved_indication":"Epilepsy / Bipolar Disorder","mechanism":"HDAC inhibitor / GABA transaminase inhibitor","clinical_phase":"Approved (1967)","protein_targets":["HDAC1","HDAC2","ABAT","CACNA1A","SCN1A","GSK3B"],"display_targets":["HDAC1","HDAC2","ABAT","CACNA1A","SCN1A"],"chembl_id":"CHEMBL109"},
    "Lithium":            {"approved_indication":"Bipolar Disorder","mechanism":"GSK-3β inhibitor / inositol monophosphatase inhibitor","clinical_phase":"Approved (1970)","protein_targets":["GSK3B","GSK3A","INPP1","BCL2","TP53","BDNF"],"display_targets":["GSK3B","GSK3A","INPP1","BCL2","BDNF"],"chembl_id":"CHEMBL1401"},
    "Methotrexate":       {"approved_indication":"Rheumatoid Arthritis / Cancer","mechanism":"DHFR inhibitor / anti-folate","clinical_phase":"Approved (1953)","protein_targets":["DHFR","TYMS","ATIC","GART","ABCG2","SLC19A1"],"display_targets":["DHFR","TYMS","ATIC","GART","ABCG2"],"chembl_id":"CHEMBL34259"},
    "Hydroxychloroquine": {"approved_indication":"Malaria / Lupus / RA","mechanism":"TLR signaling inhibitor / lysosomal pH modifier","clinical_phase":"Approved (1955)","protein_targets":["TLR7","TLR9","CLCN7","ACE2","TNF","IL6"],"display_targets":["TLR7","TLR9","CLCN7","ACE2","TNF"],"chembl_id":"CHEMBL1370"},
    "Dexamethasone":      {"approved_indication":"Inflammatory & Autoimmune / COVID-19","mechanism":"Glucocorticoid receptor agonist","clinical_phase":"Approved (1958)","protein_targets":["NR3C1","NFKB1","IL6","TNF","PTGS2","ANXA1"],"display_targets":["NR3C1","NFKB1","IL6","TNF","PTGS2"],"chembl_id":"CHEMBL535"},
    "Resveratrol":        {"approved_indication":"Investigational / Nutraceutical","mechanism":"SIRT1 activator / NF-kB inhibitor","clinical_phase":"Phase II / Investigational","protein_targets":["SIRT1","NFKB1","PTGS1","PTGS2","TP53","AKT1"],"display_targets":["SIRT1","NFKB1","PTGS1","PTGS2","TP53"],"chembl_id":"CHEMBL449437"},
    "Curcumin":           {"approved_indication":"Investigational / Nutraceutical","mechanism":"NF-kB inhibitor / anti-inflammatory / antioxidant","clinical_phase":"Phase II / Investigational","protein_targets":["NFKB1","TP53","VEGFA","EGFR","AKT1","BCL2"],"display_targets":["NFKB1","TP53","VEGFA","EGFR","AKT1"],"chembl_id":"CHEMBL112"},
}

# ---------------------------------------------------------------------------
# DISEASE MAP
# ---------------------------------------------------------------------------
DISEASE_MAP: Dict[str, Dict] = {
    "Colorectal Cancer":               {"cat":"Oncology",      "genes":["KRAS","APC","TP53","SMAD4","BRAF","PIK3CA","PTEN","MLH1","MTOR"],           "trials":["NCT02042092","NCT01864681","NCT01312467"],"disgenet_id":"C0009402"},
    "Pancreatic Cancer":               {"cat":"Oncology",      "genes":["KRAS","TP53","SMAD4","CDKN2A","BRCA2","MTOR","ARID1A"],                     "trials":["NCT02359058"],                           "disgenet_id":"C0235974"},
    "Breast Cancer":                   {"cat":"Oncology",      "genes":["BRCA1","BRCA2","TP53","PIK3CA","ERBB2","ESR1","PTEN","MTOR"],                "trials":[],                                        "disgenet_id":"C0006142"},
    "Ovarian Cancer":                  {"cat":"Oncology",      "genes":["TP53","BRCA1","BRCA2","PTEN","PIK3CA","KRAS","MTOR"],                        "trials":[],                                        "disgenet_id":"C0029925"},
    "Lung Cancer (NSCLC)":             {"cat":"Oncology",      "genes":["TP53","KRAS","EGFR","ALK","BRAF","PIK3CA","STK11","RB1"],                    "trials":[],                                        "disgenet_id":"C0684249"},
    "Glioblastoma":                    {"cat":"Oncology",      "genes":["TP53","PTEN","EGFR","IDH1","CDKN2A","RB1","MTOR"],                           "trials":[],                                        "disgenet_id":"C0017636"},
    "Prostate Cancer":                 {"cat":"Oncology",      "genes":["AR","TP53","PTEN","RB1","MYC","ERG","MTOR"],                                 "trials":[],                                        "disgenet_id":"C0376358"},
    "Melanoma":                        {"cat":"Oncology",      "genes":["BRAF","NRAS","PTEN","TP53","CDKN2A","KIT"],                                  "trials":[],                                        "disgenet_id":"C0025202"},
    "Hepatocellular Carcinoma":        {"cat":"Oncology",      "genes":["TP53","CTNNB1","ARID1A","RB1","AXIN1","MTOR","IGF1R"],                       "trials":[],                                        "disgenet_id":"C2239176"},
    "Acute Myeloid Leukemia":          {"cat":"Oncology",      "genes":["FLT3","NPM1","DNMT3A","IDH1","IDH2","TP53","RUNX1","KIT"],                   "trials":[],                                        "disgenet_id":"C0023467"},
    "Chronic Lymphocytic Leukemia":    {"cat":"Oncology",      "genes":["TP53","ATM","NOTCH1","SF3B1","BIRC3","MYD88","BCL2"],                        "trials":[],                                        "disgenet_id":"C0023434"},
    "Multiple Myeloma":                {"cat":"Oncology",      "genes":["CRBN","IKZF1","IKZF3","TP53","RB1","MYC","CCND1"],                           "trials":[],                                        "disgenet_id":"C0026764"},
    "Diffuse Large B-Cell Lymphoma":   {"cat":"Oncology",      "genes":["MYC","BCL2","BCL6","TP53","CD79B","MYD88","NFKB1"],                          "trials":[],                                        "disgenet_id":"C0079744"},
    "Bladder Cancer":                  {"cat":"Oncology",      "genes":["TP53","FGFR3","CDKN2A","RB1","PIK3CA","ERBB2","MTOR"],                       "trials":[],                                        "disgenet_id":"C0005684"},
    "Renal Cell Carcinoma":            {"cat":"Oncology",      "genes":["VHL","PBRM1","BAP1","SETD2","KDM5C","MTOR","KIT"],                           "trials":[],                                        "disgenet_id":"C0007134"},
    "Thyroid Cancer":                  {"cat":"Oncology",      "genes":["BRAF","RAS","RET","TP53","PIK3CA","PTEN"],                                   "trials":[],                                        "disgenet_id":"C0007115"},
    "Gastric Cancer":                  {"cat":"Oncology",      "genes":["TP53","KRAS","ERBB2","CDH1","ARID1A","PIK3CA","MTOR"],                       "trials":[],                                        "disgenet_id":"C0024623"},
    "Cervical Cancer":                 {"cat":"Oncology",      "genes":["TP53","RB1","PIK3CA","KRAS","STK11","ERBB2"],                                "trials":[],                                        "disgenet_id":"C0007867"},
    "Esophageal Cancer":               {"cat":"Oncology",      "genes":["TP53","CDKN2A","ERBB2","EGFR","VEGFA","PIK3CA"],                             "trials":[],                                        "disgenet_id":"C0030319"},
    "Alzheimer's Disease":             {"cat":"Neurology",     "genes":["APP","PSEN1","PSEN2","APOE","MAPT","TREM2","MTOR","CLU"],                    "trials":["NCT04042259"],                           "disgenet_id":"C0002395"},
    "Parkinson's Disease":             {"cat":"Neurology",     "genes":["SNCA","LRRK2","PINK1","PRKN","GBA","UCHL1","MTOR"],                          "trials":[],                                        "disgenet_id":"C0030567"},
    "ALS":                             {"cat":"Neurology",     "genes":["SOD1","FUS","TARDBP","ALS2","SETX","MTOR","NEFL"],                           "trials":[],                                        "disgenet_id":"C0002736"},
    "Multiple Sclerosis":              {"cat":"Neurology",     "genes":["HLA-DRB1","IL7R","TNFRSF1A","IRF8","CD58","MTOR"],                          "trials":[],                                        "disgenet_id":"C0026769"},
    "Epilepsy":                        {"cat":"Neurology",     "genes":["SCN1A","SCN2A","KCNQ2","CDKL5","PCDH19","TSC1","MTOR"],                     "trials":[],                                        "disgenet_id":"C0014544"},
    "Huntington's Disease":            {"cat":"Neurology",     "genes":["HTT","BDNF","PGC1A","SIRT1","HDAC4","CASP3","GSK3B"],                       "trials":[],                                        "disgenet_id":"C0020179"},
    "Schizophrenia":                   {"cat":"Neurology",     "genes":["DISC1","COMT","DTNBP1","NRG1","DAOA","DRD2","AKT1"],                        "trials":[],                                        "disgenet_id":"C0036341"},
    "Major Depression":                {"cat":"Neurology",     "genes":["SLC6A4","BDNF","FKBP5","CRHR1","HTR2A","MAOA","GSK3B"],                     "trials":[],                                        "disgenet_id":"C0011570"},
    "Bipolar Disorder":                {"cat":"Neurology",     "genes":["CACNA1C","ANK3","ODZ4","NCAN","GSK3B","ADCY2","HDAC1"],                      "trials":[],                                        "disgenet_id":"C0005586"},
    "Autism Spectrum Disorder":        {"cat":"Neurology",     "genes":["SHANK3","NLGN3","NRXN1","TSC1","PTEN","MECP2","MTOR"],                      "trials":[],                                        "disgenet_id":"C0004352"},
    "Polycystic Ovary Syndrome":       {"cat":"Metabolic",     "genes":["INSR","PPARG","CYP11A1","LHCGR","IGF1R","PRKAA1"],                          "trials":[],                                        "disgenet_id":"C0032460"},
    "Non-Alcoholic Fatty Liver":       {"cat":"Metabolic",     "genes":["PNPLA3","TM6SF2","GCKR","PPARG","ADIPOQ","SIRT1"],                          "trials":["NCT03432871"],                           "disgenet_id":"C0400966"},
    "Obesity":                         {"cat":"Metabolic",     "genes":["LEP","LEPR","MC4R","FTO","POMC","PCSK1","PPARG","SIRT1"],                    "trials":[],                                        "disgenet_id":"C0028754"},
    "Type 1 Diabetes":                 {"cat":"Metabolic",     "genes":["INS","PTPN22","HLA-DQB1","CTLA4","IL2RA","IFIH1"],                           "trials":[],                                        "disgenet_id":"C0011854"},
    "Metabolic Syndrome":              {"cat":"Metabolic",     "genes":["PPARG","ADIPOQ","LEP","TNF","RETN","IL6","PRKAA1"],                          "trials":[],                                        "disgenet_id":"C0524620"},
    "Gout":                            {"cat":"Metabolic",     "genes":["SLC2A9","ABCG2","SLC22A12","XDH","NLRP3","IL1B"],                            "trials":[],                                        "disgenet_id":"C0018099"},
    "Osteoporosis":                    {"cat":"Metabolic",     "genes":["LRP5","TNFSF11","TNFRSF11B","VDR","ESR1","SOST"],                           "trials":[],                                        "disgenet_id":"C0029456"},
    "Heart Failure":                   {"cat":"Cardiovascular","genes":["ACE","ADRB1","ADRB2","MYH7","TNNT2","SCN5A","PPARG"],                       "trials":[],                                        "disgenet_id":"C0018801"},
    "Hypertension":                    {"cat":"Cardiovascular","genes":["ACE","AGT","AGTR1","KCNMB1","GNB3","NOS3","PPARG"],                         "trials":[],                                        "disgenet_id":"C0020538"},
    "Atrial Fibrillation":             {"cat":"Cardiovascular","genes":["KCNQ1","KCNH2","SCN5A","GJA5","PITX2","ZFHX3"],                            "trials":[],                                        "disgenet_id":"C0004238"},
    "Rheumatoid Arthritis":            {"cat":"Cardiovascular","genes":["TNF","IL6","IL1B","PTPN22","STAT4","CTLA4","TP53"],                         "trials":[],                                        "disgenet_id":"C0003873"},
    "Coronary Artery Disease":         {"cat":"Cardiovascular","genes":["PCSK9","LDLR","APOE","HMGCR","LPL","CETP","NOS3"],                          "trials":[],                                        "disgenet_id":"C0010068"},
    "Stroke":                          {"cat":"Cardiovascular","genes":["F5","F2","ACE","APOE","MTHFR","NOS3","PTGS1"],                               "trials":[],                                        "disgenet_id":"C0038454"},
    "Pulmonary Arterial Hypertension": {"cat":"Cardiovascular","genes":["BMPR2","ACVRL1","ENG","SMAD9","CAV1","KCNK3","NOS3"],                       "trials":[],                                        "disgenet_id":"C0152171"},
    "Chronic Kidney Disease":          {"cat":"Cardiovascular","genes":["ACE","AGT","NPHS1","NPHS2","UMOD","MUC1","PPARG"],                          "trials":[],                                        "disgenet_id":"C1561643"},
    "Systemic Lupus Erythematosus":    {"cat":"Cardiovascular","genes":["TREX1","IRF5","STAT4","BLK","ITGAM","SPP1","NFKB1"],                        "trials":[],                                        "disgenet_id":"C0024141"},
    "Inflammatory Bowel Disease":      {"cat":"Cardiovascular","genes":["NOD2","ATG16L1","IL23R","IRGM","PTGER4","LRRK2","TNF"],                     "trials":[],                                        "disgenet_id":"C0021390"},
    "COVID-19":                        {"cat":"Infectious",    "genes":["ACE2","TMPRSS2","FURIN","IL6","TNF","IFNAR1","NFKB1"],                       "trials":[],                                        "disgenet_id":"C5203670"},
    "HIV/AIDS":                        {"cat":"Infectious",    "genes":["CCR5","CXCR4","CD4","HLA-B","APOBEC3G","BST2","TNF"],                       "trials":[],                                        "disgenet_id":"C0001175"},
    "Tuberculosis":                    {"cat":"Infectious",    "genes":["VDR","SLC11A1","TLR2","TNF","IL12B","NOS2","NFKB1"],                        "trials":[],                                        "disgenet_id":"C0041296"},
    "Malaria":                         {"cat":"Infectious",    "genes":["HBB","G6PD","CYP2C8","FCGR2B","DARC","TNF","IL6"],                          "trials":[],                                        "disgenet_id":"C0024530"},
    "Hepatitis C":                     {"cat":"Infectious",    "genes":["IFNL3","IL28B","LDLR","SCARB1","CD81","CLDN1","TNF"],                       "trials":[],                                        "disgenet_id":"C0220847"},
    "Cystic Fibrosis":                 {"cat":"Rare",          "genes":["CFTR","SLC6A14","TGFB1","EDNRA","MBL2","NFKB1"],                            "trials":[],                                        "disgenet_id":"C0010674"},
    "Duchenne Muscular Dystrophy":     {"cat":"Rare",          "genes":["DMD","UTRN","SSPN","LTBP4","TGFB1","MTOR"],                                "trials":[],                                        "disgenet_id":"C0013144"},
    "Sickle Cell Disease":             {"cat":"Rare",          "genes":["HBB","HBG1","HBG2","BCL11A","KLF1","HMOX1"],                               "trials":[],                                        "disgenet_id":"C0002895"},
}

CATEGORIES = ["All categories","Oncology","Neurology","Metabolic","Cardiovascular","Infectious","Rare"]

# ---------------------------------------------------------------------------
# AI RATIONALES  (static for selected pairs)
# ---------------------------------------------------------------------------
AI_RATIONALES = {
    ("Metformin","Colorectal Cancer"): {
        "text":"Metformin activates AMPK, suppressing mTOR — a central driver of colorectal cancer cell proliferation. The AMPK-mTOR axis intersects the PI3K/AKT pathway frequently mutated in CRC. Twelve observational studies show 25-37% lower CRC incidence with metformin use.",
        "risks":[("orange","Generic drug — limited IP protection"),("orange","GI side effects may limit dosing in frail patients"),("green","Favorable safety record reduces regulatory risk")],
        "rec_label":"Strong pursue","rec_body":"Phase III trial active. Clear mechanism via AMPK/mTOR. 12+ observational studies. Consider partnership or IND filing for combination therapy.",
    },
    ("Metformin","Polycystic Ovary Syndrome"): {
        "text":"Metformin's insulin-sensitizing effect via AMPK directly addresses PCOS-driven hyperinsulinemia. IGF1R and PPARG overlap provides a strong mechanistic anchor. Multiple RCTs confirm restoration of menstrual cycles and reduced androgen levels.",
        "risks":[("green","Already used off-label — de-risked from safety standpoint"),("orange","May face reimbursement hurdles due to overlap with T2D indication")],
        "rec_label":"Strong pursue","rec_body":"Well-established clinical practice supports pursuing formal indication. Focus on adolescent PCOS as an underserved sub-population.",
    },
    ("Aspirin","Colorectal Cancer"): {
        "text":"Aspirin's COX-2 inhibition reduces prostaglandin E2, a known promoter of colorectal carcinogenesis. VEGF suppression additionally limits tumor angiogenesis. Large cohort studies consistently show 25-30% reduction in CRC incidence with regular aspirin use.",
        "risks":[("orange","GI bleeding risk limits use in elderly patients"),("green","Widely available, low cost, well-understood safety profile")],
        "rec_label":"Strong pursue","rec_body":"Robust epidemiological evidence supports CRC prevention indication. Consider low-dose aspirin trial in high-risk populations.",
    },
}

# ---------------------------------------------------------------------------
# DISEASE PREVALENCE (US, Feature 23 – regulatory pathway)
# ---------------------------------------------------------------------------
DISEASE_PREVALENCE: Dict[str, int] = {
    "Colorectal Cancer": 1_500_000, "Pancreatic Cancer": 60_000, "Breast Cancer": 3_800_000,
    "Ovarian Cancer": 220_000, "Lung Cancer (NSCLC)": 550_000, "Glioblastoma": 13_000,
    "Prostate Cancer": 3_300_000, "Melanoma": 1_000_000, "Hepatocellular Carcinoma": 42_000,
    "Acute Myeloid Leukemia": 20_000, "Chronic Lymphocytic Leukemia": 200_000, "Multiple Myeloma": 160_000,
    "Diffuse Large B-Cell Lymphoma": 80_000, "Bladder Cancer": 700_000, "Renal Cell Carcinoma": 450_000,
    "Thyroid Cancer": 900_000, "Gastric Cancer": 95_000, "Cervical Cancer": 250_000,
    "Esophageal Cancer": 45_000, "Alzheimer's Disease": 6_500_000, "Parkinson's Disease": 1_000_000,
    "ALS": 30_000, "Multiple Sclerosis": 1_000_000, "Epilepsy": 3_400_000,
    "Huntington's Disease": 30_000, "Schizophrenia": 3_500_000, "Major Depression": 21_000_000,
    "Bipolar Disorder": 5_700_000, "Autism Spectrum Disorder": 5_400_000,
    "Polycystic Ovary Syndrome": 5_000_000, "Non-Alcoholic Fatty Liver": 100_000_000,
    "Obesity": 100_000_000, "Type 1 Diabetes": 1_600_000, "Metabolic Syndrome": 75_000_000,
    "Gout": 9_200_000, "Osteoporosis": 10_000_000, "Heart Failure": 6_200_000,
    "Hypertension": 119_000_000, "Atrial Fibrillation": 6_100_000, "Rheumatoid Arthritis": 1_300_000,
    "Coronary Artery Disease": 20_000_000, "Stroke": 7_000_000, "Pulmonary Arterial Hypertension": 50_000,
    "Chronic Kidney Disease": 37_000_000, "Systemic Lupus Erythematosus": 1_500_000,
    "Inflammatory Bowel Disease": 3_100_000, "COVID-19": 100_000_000, "HIV/AIDS": 1_200_000,
    "Tuberculosis": 13_000_000, "Malaria": 240_000_000, "Hepatitis C": 2_400_000,
    "Cystic Fibrosis": 40_000, "Duchenne Muscular Dystrophy": 15_000, "Sickle Cell Disease": 100_000,
}

# ---------------------------------------------------------------------------
# PATIENT SUBGROUPS  (Feature 22)
# ---------------------------------------------------------------------------
PATIENT_SUBGROUPS: Dict[str, List[Dict]] = {
    "Colorectal Cancer": [
        {"subgroup":"KRAS wild-type","biomarker":"KRAS WT","predicted_response":"High","evidence":"RCT data (CRYSTAL trial)"},
        {"subgroup":"MSI-High","biomarker":"MMR-deficient","predicted_response":"High","evidence":"FDA approval (pembrolizumab)"},
        {"subgroup":"BRAF V600E mutant","biomarker":"BRAF V600E","predicted_response":"Low","evidence":"Retrospective cohort"},
        {"subgroup":"HER2 amplified","biomarker":"ERBB2 amp","predicted_response":"Medium","evidence":"Phase II data"},
    ],
    "Alzheimer's Disease": [
        {"subgroup":"APOE ε4 carriers","biomarker":"APOE4","predicted_response":"Medium","evidence":"Observational"},
        {"subgroup":"Early-onset (<65)","biomarker":"APP/PSEN1 mutation","predicted_response":"Low","evidence":"Case series"},
        {"subgroup":"Amyloid PET positive","biomarker":"Aβ PET+","predicted_response":"High","evidence":"Phase III (aducanumab)"},
    ],
    "Breast Cancer": [
        {"subgroup":"ER+ / HER2-","biomarker":"ESR1+, ERBB2-","predicted_response":"High","evidence":"ATAC trial"},
        {"subgroup":"Triple negative","biomarker":"ER-/PR-/HER2-","predicted_response":"Low","evidence":"Meta-analysis"},
        {"subgroup":"BRCA1/2 mutant","biomarker":"BRCA1/2 mut","predicted_response":"High","evidence":"PARP inhibitor trials"},
    ],
}

# ---------------------------------------------------------------------------
# REGULATORY LINKS  (Feature 23)
# ---------------------------------------------------------------------------
REG_LINKS = {
    "Orphan Drug Designation": "https://www.fda.gov/patients/rare-diseases-fda/orphan-drug-product-designation",
    "Fast Track":               "https://www.fda.gov/patients/fast-track-breakthrough-therapy-accelerated-approval-priority-review/fast-track",
    "Breakthrough Therapy":     "https://www.fda.gov/patients/fast-track-breakthrough-therapy-accelerated-approval-priority-review/breakthrough-therapy",
    "Accelerated Approval":     "https://www.fda.gov/patients/fast-track-breakthrough-therapy-accelerated-approval-priority-review/accelerated-approval",
}

# ---------------------------------------------------------------------------
# SERIOUS ADVERSE EVENT TERMS  (Feature 20)
# ---------------------------------------------------------------------------
SERIOUS_AE_TERMS = {
    "cardiac","death","hepatic","renal","nephrotoxicity","cardiotoxicity",
    "hepatotoxicity","liver failure","kidney failure","cardiac arrest",
    "myocardial infarction","stroke","fatal","anaphylaxis","agranulocytosis",
}
