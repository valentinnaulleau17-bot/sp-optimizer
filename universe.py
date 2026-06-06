# ============================================================
# universe.py — Complete Global Equity Universe
# ============================================================
# Sources: S&P 500 (503), STOXX 600 by country, FTSE 100,
#          Nikkei 225 (top), Hang Seng, EM large-caps
# Total: ~1 200 tickers across 8 geographic regions
# All tickers use yfinance convention (e.g. .PA, .L, .DE …)
# ============================================================

# ── AMERICAS ─────────────────────────────────────────────────

# S&P 500 — full constituent list (503 tickers)
SP500 = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE",
    "ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG",
    "AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL",
    "ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL",
    "BAC","BK","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BMY",
    "AVGO","BR","BRO","BF-B","BLDR","BSX","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX",
    "CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL",
    "SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME",
    "CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CPAY",
    "CTVA","CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHR","DHI","DRI","DVA","DAY",
    "DECK","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV",
    "DOW","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH",
    "ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES","EXC",
    "EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR",
    "FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV",
    "GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC",
    "HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN",
    "HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU",
    "ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K",
    "KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW",
    "LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC",
    "MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET",
    "MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST",
    "MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI",
    "NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL",
    "OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR",
    "PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD",
    "PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG",
    "REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB",
    "STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX",
    "STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP",
    "TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV",
    "TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS",
    "VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW",
    "WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WM","WY",
    "WHR","WMB","WTW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZION","ZTS",
    # Additional S&P 500 names
    "ABNB","ALGN","AXON","BLDR","CDAY","CEG","CPAY","CRWD","CSGP","DAY","DVA","ENPH",
    "EG","EPAM","EXR","FFIV","FTNT","GEV","GNRC","HOLX","INVH","JNPR","KVUE","LDOS",
    "LKQ","LYV","MKTX","MOH","MPWR","MRNA","MTCH","NCLH","NRG","NXPI","ODFL","PANW",
    "PAYC","POOL","PODD","QRVO","RVTY","SMCI","SOLV","TRMB","ULTA","VLTO","VST","WBD","ZBRA",
]

# S&P 400 (Mid Cap) — representative selection
SP400 = [
    "ACHC","ACI","AGIO","AIT","AL","ALKS","AMG","AMKR","AMSF","AMWD","ANET","AOSL","APLS",
    "ARCH","AROC","ASB","ATI","AVA","AVNT","AWI","BANF","BCC","BECN","BFH","BKH","BPMC",
    "BRC","BRKR","CABO","CALM","CASH","CATY","CBD","CBT","CCSI","CENTA","CIR","CLH","CMP",
    "CNO","COLB","CRI","CRK","CROX","CSL","CTB","CUTR","CWK","DINO","DLB","DORM","DUOL",
    "DXC","EFC","ENSG","ENS","ENV","EPRT","ESAB","ESNT","ESE","EVH","EWBC","FAF","FBIN",
    "FHN","FIVE","FNB","FNF","FR","FUL","GEF","GNTX","GPK","GTLS","HAE","HBI","HGV",
    "HHH","HIW","HLI","HNI","HOLO","HUN","IAA","IBP","IDCC","INDB","INGR","IOSP","ITRI",
    "JACK","JBT","JHG","JELD","JXN","KAI","KFRC","KN","KNF","KSS","KTB","LAMR","LGF-A",
    "LGIH","LMAT","LNTH","LQDT","LSTR","LTC","MAIN","MAN","MANT","MATX","MEDP","MELI",
    "MEG","MIDD","MLI","MMSI","MMS","MNR","MOG-A","MRTN","MSGS","MTN","MTX","MUR","NBR",
    "NEOG","NEX","NFG","NJR","NNN","NOV","NOVT","NVT","OGE","OGS","OHI","OLN","OMCL",
    "ORI","OUT","PACW","PBF","PBI","PCH","PDCO","PII","PINC","PLTK","PLT","PNM","POWI",
    "PRGO","PSB","RBC","RDN","RGEN","RHP","RITM","RLI","RMBS","RNR","ROCC","RPM","RRX",
    "RSG","RUSHA","RYAN","SCI","SIGI","SLG","SM","SMP","SNV","SPSC","SSD","STEL","STNG",
    "SWTX","SXC","TALO","TDC","THO","TKR","TMHC","TNL","TNET","TOWN","TPH","TRN","TRNO",
    "TRMK","TSEM","TTGT","TUP","TXRH","UFI","UGI","UMBF","UMPQ","UNF","UNVR","UONE",
    "UPWK","URBN","VCEL","VFC","VIAD","VIAV","VLY","VMI","VNO","VSCO","VSH","VVV","WAL",
    "WBS","WDFC","WERN","WEX","WGO","WMS","WOR","WRE","WTBA","WTRE","XNCR","XPO","Y","ZWS",
]

# S&P 600 (Small Cap) — representative selection
SP600 = [
    "ABM","ACAD","ACLS","ACRX","ADUS","AEO","AFG","AGM","AGYS","AHH","AIN","AJRD","ALEX",
    "ALGT","ALRM","AMAG","AMBC","AMPH","AMRC","AMRS","AMWD","AN","ANDE","ANF","AOBC","APEI",
    "APTS","ARC","ARCB","ARCW","ARKG","ARKQ","ARL","ARMO","AROW","ARQT","ASO","ASTE",
    "ATEX","ATNI","ATRI","ATRO","ATSG","AUB","AVAV","AVNS","AWR","AZZ","BANR","BBSI",
    "BCEI","BCO","BCOR","BDC","BFS","BG","BGFV","BHLB","BKD","BKNG","BKU","BLKB","BLMN",
    "BNED","BNS","BOH","BOOT","BPOP","BRC","BRKL","BRKS","BSIG","BSRR","BV","BXS","BYFC",
    "CABO","CACC","CAKE","CBTX","CBUS","CCMP","CEIX","CELH","CENT","CENX","CFFI","CGNX",
    "CHCO","CHGG","CHRS","CINR","CIVB","CLAR","CLB","CLDT","CLFD","CLNE","CLPS","CLXT",
    "CMCO","CMRE","CMRX","CNMD","CNOB","CNSL","COHU","COMT","CONN","CPSI","CPSS","CRESY",
    "CREV","CRIS","CRLBF","CRNC","CRNT","CRVL","CSGP","CTBI","CTG","CTLP","CTO","CVLY",
    "CVLT","CWD","CWEN","DCOM","DFIN","DFPH","DGII","DLHC","DLNG","DNOW","DOCN","DOMO",
    "DRQ","DSGX","DTIL","EAF","EBC","EGAN","ELME","EMCF","EML","ENOV","ENVA","ENV","EPAC",
    "EPZM","ESCA","ESGR","ESTU","ETWO","EVBG","EVERI","EVGO","EXK","EXLS","EXTN","EZPW",
    "FARO","FBK","FBMS","FBSS","FCNCA","FCPH","FCRD","FCT","FDEF","FDP","FELE","FFBC",
    "FFIN","FG","FGEN","FISI","FITBI","FLGT","FLIR","FLNC","FMBH","FMBI","FMNB","FMTX",
    "FORR","FRME","FRPH","FRUN","FSB","FSBW","FSCO","FULT","FUL","FUNC","FUSN","GBCI",
    "GCO","GFF","GFED","GGAL","GHL","GILT","GKOS","GLDD","GLNG","GLOG","GLP","GLRE",
    "GMAB","GMRE","GNSS","GOE","GOLF","GPMT","GPX","GRFS","GRM","GSIT","GTHX","GTIM",
    "HAYN","HBB","HBIO","HCAT","HCI","HCKT","HCSG","HERO","HEXO","HFS","HIBB","HIE",
    "HIFS","HMST","HOFT","HOLI","HOLX","HONE","HOUS","HQY","HSC","HTBK","HTH","HUBG",
    "HVT","HWC","HWKN","HYLN","HZO","ICUI","IGT","IIIN","IIPR","IIVI","IMKTA","IMVT",
    "INBK","INDB","INFI","INFN","INFU","IPAR","IPHI","IRDM","IROQ","IRWD","ISBC","ISSC",
    "JACK","JBSS","JELD","JNCE","JOUT","JRVR","JSPR","KALA","KALU","KAR","KBAL","KBSF",
    "KE","KEQU","KFRC","KFY","KLIC","KMT","KN","KNTK","KOP","KRYS","KSPN","KTOS","KVHI",
    "LARK","LBAI","LBC","LBND","LCI","LDI","LEGH","LGND","LHCG","LIQT","LITE","LIVN",
    "LJPC","LKQ","LMB","LMNR","LMT","LNDC","LNTH","LOAN","LORL","LPLA","LPSN","LQDA",
    "LQDT","LRN","LSCC","LSTR","LTBR","LTC","LTRE","LUCK","LUNR","LYTS","MANT","MATW",
    "MATX","MBI","MBUU","MBWM","MCBC","MCHP","MCRB","MDRX","MEDP","MESA","MESG","MFC",
    "MFG","MGLN","MGNX","MGRC","MIST","MKSI","MMAC","MMSI","MNRO","MODN","MOFG","MOGO",
    "MORN","MRC","MRCY","MRLN","MRVL","MSEX","MSTR","MTH","MTSC","MUSA","NBTB","NBTX",
    "NCAT","NEIE","NEOG","NET","NEWT","NFG","NHTC","NICE","NKTR","NL","NNI","NODK","NOVT",
    "NPK","NRCIA","NRIM","NSP","NTCT","NTGR","NTLS","NURO","NUS","NVRO","NXRT","NYCB",
    "NYT","OBT","OCFC","OCSL","OCUL","OESX","OFG","OGS","OIIM","OIS","OMCL","OMI","OMN",
    "ONDK","OPCH","OPTN","ORBC","ORGO","ORLY","OSBC","OSI","OSPN","OTTR","OVBC","OVID",
    "OZK","PARR","PATK","PAYA","PBFS","PBHC","PBTL","PBYI","PCTY","PEBO","PFIS","PFNL",
    "PGC","PGNY","PHII","PHR","PINC","PIXY","PKOH","PKX","PLM","PLMR","PLNT","PLSE",
    "PLXP","PMFG","PMTS","PNM","PNTM","POAI","POOL","PORF","POWL","PRDO","PRGS","PRLB",
    "PRO","PRTA","PRVB","PRXL","PSMT","PSTL","PTCT","PTI","PTLO","PTSI","PUMP","PVBC",
    "PWOD","PWSC","QDEL","QGEN","QLI","QLYS","QNST","QUOT","RBBA","RBCAA","RBCN","RBZ",
    "RCII","RCKY","RCM","RCON","RDUS","REX","REXR","RFIL","RGEN","RGLD","RICK","RIGL",
    "RIVN","RLAY","RLMD","RMR","RMTI","ROAD","ROCH","ROCR","ROIC","ROLL","RPTX","RPXC",
    "RRGB","RRTS","RTLR","RUN","RUSHA","RUTH","RXMD","SASR","SAVA","SBH","SBSI","SCHL",
    "SCSC","SCTX","SEIC","SFBC","SFLY","SGEN","SGH","SGMS","SGMO","SHBI","SHO","SHYF",
    "SIC","SIFI","SIMU","SITO","SKYW","SLCA","SLP","SMBC","SMCP","SMBK","SMCI","SMDM",
    "SMG","SMP","SMPL","SNBR","SNCR","SONO","SPFI","SPKE","SPNV","SPOT","SPSC","SPWH",
    "SQZ","SRCE","SRI","SRLP","SRPT","SSTI","STBA","STFC","STGW","STLD","STLY","STRM",
    "STRS","STRA","SUNS","SUPN","SURG","SWI","SXC","SYBT","SYKE","TALO","TBBK","TBIO",
    "TBNK","TCBIP","TCMD","TCPIP","TDOC","TENB","TGTX","THQ","THTX","TIG","TIS","TITN",
    "TLRA","TLYS","TMDX","TMST","TNAT","TORC","TOUR","TOWN","TPVG","TPVY","TRDA","TRIL",
    "TRNO","TRNX","TRUP","TSRI","TTGT","TUSK","TVTX","TWKS","TZOO","UHAL","UNAM","UNFI",
    "UNIT","URGN","USAK","USAP","USCR","UTHR","UVSP","UWHR","VBTX","VCNX","VCYT","VFIN",
    "VGR","VIBI","VICR","VIRT","VIV","VLGEA","VMEO","VNDA","VNOM","VNRX","VOC","VONE",
    "VRRM","VSEC","WAFD","WASH","WBT","WD","WDFC","WEST","WETF","WFRD","WINA","WKHS",
    "WMGI","WNEB","WNS","WOOD","WSBF","WSBK","WSFS","WTS","WTTR","WULF","WWW","XBIT",
    "XCUR","XERS","XLNX","XOMA","XPEL","XPER","YEXT","YLMN","YORW","YRCW","ZGN","ZION",
]

# Canada — TSX 60
TSX60 = [
    "RY.TO","TD.TO","ENB.TO","CNR.TO","BNS.TO","BMO.TO","TRP.TO","CP.TO","MFC.TO","SU.TO",
    "CM.TO","ABX.TO","BCE.TO","CVE.TO","DOL.TO","GIB-A.TO","L.TO","MG.TO","NA.TO","NTR.TO",
    "PBR-A.TO","QSR.TO","REI-UN.TO","RCI-B.TO","SAP.TO","SJR-B.TO","SHOP.TO","SNC.TO",
    "TRI.TO","TRQ.TO","WCN.TO","WN.TO","WSP.TO","AEM.TO","AGI.TO","AQN.TO","ATD.TO",
    "BEI-UN.TO","BIP-UN.TO","CAE.TO","CCO.TO","CHP-UN.TO","CTC-A.TO","EMA.TO","FNV.TO",
    "FTS.TO","GWO.TO","H.TO","IAG.TO","IFC.TO","IMO.TO","K.TO","KEY.TO","KXS.TO","NFI.TO",
    "POW.TO","PPL.TO","PWF.TO","RBA.TO","SLF.TO","TECK-B.TO","TOU.TO","WFG.TO",
]

# Brazil — Bovespa top 40
BOVESPA = [
    "VALE3.SA","PETR4.SA","ITUB4.SA","BBDC4.SA","ABEV3.SA","B3SA3.SA","WEGE3.SA","RENT3.SA",
    "LREN3.SA","EMBR3.SA","BBAS3.SA","JBSS3.SA","EQTL3.SA","MGLU3.SA","SUZB3.SA","RADL3.SA",
    "HYPE3.SA","TOTS3.SA","CSNA3.SA","BRAP4.SA","CSAN3.SA","CPFE3.SA","ECOR3.SA","ELET3.SA",
    "FLRY3.SA","GNDI3.SA","GOAU4.SA","HAPV3.SA","IRBR3.SA","ITSA4.SA","KLBN11.SA","MRVE3.SA",
    "MULT3.SA","NTCO3.SA","PETR3.SA","PRIO3.SA","QUAL3.SA","RAIL3.SA","RAIZ4.SA","SBSP3.SA",
]

# Mexico — IPC top 30
IPC_MEXICO = [
    "AMXL.MX","CEMEXCPO.MX","FEMSAUBD.MX","GMEXICOB.MX","WALMEX.MX","BIMBOA.MX",
    "TLEVISACPO.MX","GRUMAB.MX","LALAB.MX","ALSEA.MX","OMAB.MX","ASURB.MX","GAPB.MX",
    "PINFRA.MX","BOLSAA.MX","LIVEPOLC-1.MX","MEGACPO.MX","PE&OLES.MX","ORBIA.MX",
    "SITES1A.MX","CUERVO.MX","IENOVA.MX","GCARSOA1.MX","CHDRAUIB.MX","BAFAR.MX",
]

# ── FRANCE ──────────────────────────────────────────────────

# CAC 40
CAC40 = [
    "MC.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","OR.PA","RI.PA","SU.PA","DG.PA","AI.PA",
    "ACA.PA","ENGI.PA","SGO.PA","ORA.PA","VIE.PA","LR.PA","CAP.PA","BN.PA","KER.PA","PUB.PA",
    "STM.PA","HO.PA","RNO.PA","ML.PA","STLAM.PA","TEP.PA","AF.PA","LI.PA","EDEN.PA",
    "ERF.PA","BOL.PA","CA.PA","ALO.PA","SW.PA","VIV.PA","CNP.PA","ADP.PA","FP.PA",
    "CS.PA","DSY.PA",
]

# SBF 120 additional (mid caps)
SBF120_EXTRA = [
    "ABN.PA","ACA.PA","ABCA.PA","ALSTOM.PA","APAM.PA","ATL.PA","ATOS.PA","BIG.PA","BPCE.PA",
    "CBKG.PA","COFA.PA","DBV.PA","DXGE.PA","EDF.PA","EI.PA","ELIOR.PA","ESI.PA","FDJ.PA",
    "FNAC.PA","GFC.PA","GTT.PA","ILIAD.PA","INFT.PA","INNATE.PA","IPH.PA","JXR.PA","KN.PA",
    "LABO.PA","LDL.PA","LEGR.PA","MERY.PA","MMB.PA","MOM.PA","MTX.PA","NANO.PA","NEOM.PA",
    "NEXANS.PA","NRG.PA","OPT.PA","ORPEA.PA","POXEL.PA","PREC.PA","PROX.PA","RCO.PA",
    "REMY.PA","RXL.PA","SAFE.PA","SAFT.PA","SAP.PA","SCBK.PA","SESH.PA","SII.PA","SKF.PA",
    "SMCP.PA","SOBP.PA","SOITEC.PA","SPIE.PA","TEC.PA","TFII.PA","THALES.PA","TOUP.PA",
    "TXCOM.PA","UBISOFT.PA","VALEO.PA","VIRP.PA","VMX.PA","VNET.PA","WNTN.PA","XILAM.PA",
]

# ── GERMANY ─────────────────────────────────────────────────

# DAX 40
DAX40 = [
    "SAP.DE","SIE.DE","ALV.DE","MUV2.DE","DTE.DE","BMW.DE","MBG.DE","BAYN.DE","BAS.DE",
    "VOW3.DE","RWE.DE","DB1.DE","HEI.DE","IFX.DE","HEN3.DE","DHER.DE","EOAN.DE","PAH3.DE",
    "QIA.DE","ZAL.DE","ADS.DE","MTX.DE","AIXA.DE","AFX.DE","BEI.DE","EVT.DE","FRE.DE",
    "HFG.DE","HOT.DE","KGX.DE","LEG.DE","MBG.DE","NDX1.DE","P911.DE","RHM.DE","SDF.DE",
    "SHL.DE","SRT3.DE","TKA.DE","VNA.DE","WCH.DE","1COV.DE","G1A.DE","GXI.DE","HAG.DE",
]

# MDAX additional (German mid-caps)
MDAX_EXTRA = [
    "AAD.DE","AIXA.DE","ARND.DE","B5A.DE","BC8.DE","BNR.DE","CBK.DE","COP.DE","DKGR.DE",
    "DLG.DE","DMG.DE","DRW3.DE","DWNI.DE","ECK.DE","ELGX.DE","ELG.DE","EOAN.DE","EUR.DE",
    "EVD.DE","FNTN.DE","FPE3.DE","FRME.DE","GBF.DE","GCPG.DE","GLJ.DE","GYC.DE","HAB.DE",
    "HBH.DE","HDD.DE","HHFA.DE","HLE.DE","INN1.DE","INXG.DE","IOM.DE","IUR.DE","JEN.DE",
    "JUN3.DE","KBX.DE","KCO.DE","KD8.DE","KGX.DE","KPR.DE","LCY.DE","LHA.DE","LIN.DE",
    "LKPG.DE","LXS.DE","MDGR.DE","MED.DE","MEO.DE","MLP.DE","MOR.DE","MRK.DE","MVV1.DE",
    "NDA.DE","NOEJ.DE","NSU.DE","O2D.DE","OHB.DE","OSS.DE","PBBG.DE","PMOX.DE","PSM.DE",
    "PUM.DE","RAA.DE","RRTL.DE","RSL2.DE","RTL.DE","S92.DE","SAX.DE","SDAX.DE","SFQ.DE",
    "SKB.DE","SLT.DE","SNH.DE","SOBA.DE","SOW.DE","STO3.DE","SZG.DE","TBO.DE","TCO.DE",
    "TEG.DE","TLX.DE","TOM.DE","TUI1.DE","TVA.DE","UBL.DE","VBK.DE","VBSC.DE","VCK.DE",
    "VOK.DE","VOS.DE","WA8.DE","WAC.DE","WAF.DE","WBAG.DE","WIN.DE","WL6.DE","WMT.DE",
    "WUW.DE","YT1.DE","ZBRA.DE","ZEG.DE",
]

# ── UNITED KINGDOM ───────────────────────────────────────────

# FTSE 100 — complete
FTSE100 = [
    "AZN.L","SHEL.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","BATS.L","LLOY.L","BARC.L",
    "VOD.L","DGE.L","NG.L","LSEG.L","IMB.L","CPG.L","RKT.L","AAL.L","PRU.L","WPP.L",
    "GLEN.L","BHP.L","REL.L","ABF.L","AHT.L","AV.L","BA.L","BNZL.L","BRBY.L","CCH.L",
    "CNA.L","DCC.L","ENT.L","EXPN.L","FERG.L","FLTR.L","HIK.L","HL.L","IAG.L","III.L",
    "JD.L","KGF.L","LAND.L","LGEN.L","MKS.L","MNDI.L","MNG.L","NXT.L","OCDO.L","PSH.L",
    "PSN.L","PSON.L","RS1.L","SDR.L","SGE.L","SMT.L","SN.L","SPX.L","SSE.L","STAN.L",
    "SVT.L","TSCO.L","TW.L","UU.L","VTY.L","WEIR.L","WTB.L","AUTO.L","BKG.L","CRDA.L",
    "ECM.L","EZJ.L","FRES.L","GFS.L","HWDN.L","IMI.L","ITV.L","JET.L","PHNX.L","RTO.L",
    "SJP.L","TUI.L","SBRY.L","AGK.L","CPI.L","DLN.L","MTG.L","PAG.L","ADN.L","ANTO.L",
    "ASC.L","BEZ.L","BLND.L","BME.L","BRIT.L","CCC.L","DPLM.L","DRX.L","ELM.L","FLTRF.L",
    "FWONA.L","HMSO.L","INCH.L","INF.L","ITRK.L","JMAT.L","MNZS.L","MRO.L","NWG.L",
    "PETS.L","RDW.L","RMV.L","RNK.L","SAFE.L","TCNG.L","TRIG.L","WDS.L","WHR.L",
]

# FTSE 250 additional
FTSE250_EXTRA = [
    "3IN.L","ACA.L","ACSO.L","AFX.L","AGR.L","AJB.L","ALLG.L","ALPG.L","ANP.L","APAX.L",
    "ASY.L","ATG.L","ATST.L","AVST.L","AWE.L","BBOX.L","BCPT.L","BGL.L","BHMG.L","BIFF.L",
    "BOKU.L","BOWL.L","BREE.L","BSV.L","BTG.L","BWA.L","BYIT.L","CAML.L","CAMT.L","CAU.L",
    "CBG.L","CBPE.L","CCEP.L","CCK.L","CET.L","CINE.L","CLDN.L","CLG.L","CLX.L","CMH.L",
    "CMRG.L","CNN.L","COA.L","COG.L","CRCL.L","CTT.L","CVSG.L","DAN.L","DAPR.L","DBB.L",
    "DEB.L","DELT.L","DNLM.L","DOC.L","DOM.L","DSCV.L","EBOX.L","EMIS.L","ENN.L","ENQR.L",
    "EPWN.L","ESMG.L","ESN.L","ETLN.L","EVR.L","EWI.L","FALC.L","FAR.L","FCCN.L","FCH.L",
    "FDP.L","FGT.L","FIPP.L","FLO.L","FLT.L","FNX.L","FRAS.L","FSFL.L","FXI.L","GBG.L",
    "GFTU.L","GKP.L","GLE.L","GMET.L","GMG.L","GPOR.L","GRI.L","GRG.L","GSS.L","GTLY.L",
    "HAT.L","HBR.L","HDD.L","HIHO.L","HIL.L","HNT.L","HOC.L","HOTC.L","HSV.L","HTG.L",
    "HWDN.L","HWG.L","HYR.L","IGG.L","INCE.L","INM.L","INS.L","IP.L","IPBG.L","IPSA.L",
    "IRM.L","ISBA.L","ISG.L","ITM.L","ITV.L","JEL.L","JEQL.L","JFJ.L","JLEN.L","JMAT.L",
    "JMG.L","JOUL.L","JPJ.L","KAV.L","KGH.L","KIDS.L","KNOS.L","KOD.L","LAM.L","LBG.L",
    "LBE.L","LCG.L","LFS.L","LGO.L","LIO.L","LIT.L","LIV.L","LMP.L","LMIN.L","LMR.L",
    "LOOK.L","LRE.L","LRM.L","LSC.L","LWI.L","MAG.L","MAI.L","MANX.L","MBH.L","MCKS.L",
    "MCL.L","MCS.L","MER.L","MGAM.L","MGP.L","MIGO.L","MKS.L","MML.L","MNTN.L","MOTR.L",
    "MRC.L","MRL.L","MSYS.L","MTH.L","MTRE.L","MTW.L","MUT.L","MXC.L","NCB.L","NCYF.L",
    "NEX.L","NFX.L","NRR.L","NSF.L","NTOG.L","NVTA.L","NWBD.L","NWSA.L","OCI.L","OCNB.L",
    "OILB.L","OPG.L","OPTS.L","OSB.L","PAF.L","PAY.L","PCG.L","PEB.L","PET.L","PFHD.L",
    "PFD.L","PHAR.L","PIC.L","PIRT.L","PKW.L","PLC.L","PLI.L","PLZ.L","PMG.L","PMGM.L",
    "PNN.L","POG.L","PPH.L","PPR.L","PRV.L","PSQ.L","PTEC.L","QRT.L","QXL.L","RBG.L",
    "RBGP.L","RCH.L","RCN.L","REC.L","RGD.L","RGLP.L","RKH.L","RNS.L","RPC.L","RPDI.L",
    "RPP.L","RRGP.L","RSA.L","RTC.L","RTM.L","RWA.L","SAR.L","SAVE.L","SBT.L","SCE.L",
]

# ── SPAIN ────────────────────────────────────────────────────
IBEX35 = [
    "ITX.MC","SAN.MC","BBVA.MC","IBE.MC","REP.MC","TEF.MC","CLNX.MC","ACS.MC","ELE.MC",
    "GRF.MC","FER.MC","MAP.MC","MTS.MC","AENA.MC","CABK.MC","ACX.MC","AMS.MC","BKT.MC",
    "COL.MC","ENG.MC","FDR.MC","GCO.MC","IAG.MC","IDR.MC","LOG.MC","MEL.MC","MRL.MC",
    "NTGY.MC","PHM.MC","RED.MC","ROVI.MC","SAB.MC","SCYR.MC","VIS.MC","MDF.MC",
]

# ── ITALY ────────────────────────────────────────────────────
FTSE_MIB = [
    "ENI.MI","ENEL.MI","ISP.MI","UCG.MI","ATL.MI","STM.MI","MB.MI","G.MI","LDO.MI",
    "BAMI.MI","AMP.MI","CPR.MI","TRN.MI","A2A.MI","BMED.MI","BPSO.MI","FCA.MI","FBK.MI",
    "FI.MI","GEO.MI","HER.MI","IF.MI","JUVE.MI","MARR.MI","MONC.MI","MS.MI","PIRC.MI",
    "PIA.MI","PRY.MI","RACE.MI","SFER.MI","SFL.MI","SRS.MI","TOD.MI","UNI.MI","EL.MI",
    "ERG.MI","IVG.MI","PST.MI","REC.MI","TERNA.MI","TIT.MI","BPER.MI","CVAL.MI","AZM.MI",
]

# ── NETHERLANDS ──────────────────────────────────────────────
AEX = [
    "ASML.AS","HEIA.AS","NN.AS","PHIA.AS","AD.AS","AKZA.AS","INGA.AS","MT.AS","RAND.AS",
    "WKL.AS","AGN.AS","BESI.AS","IMCD.AS","LIGHT.AS","TKWY.AS","UMG.AS","VPK.AS","DSM.AS",
    "GLPG.AS","KPN.AS","OCI.AS","URW.AS","ABN.AS","ASR.AS","CCEP.AS","EXEL.AS","NSI.AS",
    "SBMO.AS","SBM.AS","TMUS.AS","UNA.AS","WDP.AS","WOLB.AS",
]

# ── SWEDEN ───────────────────────────────────────────────────
OMXS30 = [
    "VOLV-B.ST","ERIC-B.ST","SHB-A.ST","SWED-A.ST","SKA-B.ST","INVE-B.ST","ATCO-A.ST",
    "HEXA-B.ST","SAND.ST","SEB-A.ST","SSAB-A.ST","TELIA.ST","ALFA.ST","ASSA-B.ST","BOL.ST",
    "ELUX-B.ST","ESSITY-B.ST","GETI-B.ST","HM-B.ST","HUSQ-B.ST","KINV-B.ST","LATO-B.ST",
    "NDA-SE.ST","NIBE-B.ST","PEAB-B.ST","SKF-B.ST","SWMA.ST","SINCH.ST","SOBI.ST","HUFVUD-A.ST",
    "BETS-B.ST","CAST.ST","EWGM.ST","FABG.ST","FLIR.ST","HFAST-B.ST","LIFCO-B.ST","LUNE.ST",
    "NCAB.ST","NENT-B.ST","NOTE.ST","OEM-B.ST","SAAB-B.ST","SECTRA-B.ST","VNV.ST","XVIVO.ST",
]

# ── DENMARK ──────────────────────────────────────────────────
OMXC25 = [
    "NOVO-B.CO","ORSTED.CO","CARL-B.CO","COLO-B.CO","DEMANT.CO","DSV.CO","FLS.CO","GEN.CO",
    "ISS.CO","MAERSK-B.CO","NZYM-B.CO","PNDORA.CO","ROCK-B.CO","RBREW.CO","SIM.CO",
    "SYDB.CO","TRYG.CO","VWS.CO","AMBU-B.CO","BAVA.CO","CHR.CO","GMAB.CO","HLUN-B.CO",
    "NETC.CO","NDA-DK.CO","CPHG.CO","DFDS.CO","GN.CO","JYSK.CO","NORDEN.CO",
]

# ── SWITZERLAND ──────────────────────────────────────────────
SMI = [
    "ROG.SW","NESN.SW","NOVN.SW","ABBN.SW","ALC.SW","CFR.SW","GIVN.SW","HOLN.SW","KNIN.SW",
    "LONN.SW","PGHN.SW","SCMN.SW","SGSN.SW","SIKA.SW","SLHN.SW","SOON.SW","STMN.SW",
    "UBSG.SW","ZURN.SW","BALN.SW","BARN.SW","GEBN.SW","KARN.SW","LISN.SW","LOGN.SW",
    "MOBN.SW","NBEN.SW","TEMN.SW","VAKN.SW","SRENH.SW","CSGN.SW","AMS.SW","BAAR.SW",
    "BCV.SW","BEKN.SW","CICN.SW","CLN.SW","EMS.SW","HIAG.SW","HUBN.SW","INDH.SW",
    "INRN.SW","LION.SW","MBTN.SW","METN.SW","MOBN.SW","MYRN.SW","NATN.SW","NREN.SW",
    "ORON.SW","PEAN.SW","PEHN.SW","PGHN.SW","PSPN.SW","SFZN.SW","TOHN.SW","VAHN.SW","VIFN.SW",
]

# ── FINLAND ──────────────────────────────────────────────────
OMXH25 = [
    "NOKIA.HE","OUT1V.HE","NESTE.HE","FORTUM.HE","SAMPO.HE","STERV.HE","UPM.HE","WRT1V.HE",
    "KEMIRA.HE","KNEBV.HE","METSO.HE","ORNBV.HE","TELIA.HE","TIETO.HE","YTY1V.HE",
    "ACG1V.HE","AKTIA.HE","ALMA.HE","CGCBV.HE","ETTE.HE","GOFORE.HE","HUH1V.HE","ICP1V.HE",
    "KAMUX.HE","KONE.HE","LINDEX.HE","MEKKO.HE","ORNAV.HE","ORNBV.HE","PUUILO.HE","QPR1V.HE",
    "RAP1V.HE","RAP1V.HE","REXI.HE","ROBIT.HE","SRV1V.HE","STR1V.HE","TELIA1.HE",
]

# ── NORWAY ───────────────────────────────────────────────────
OBX = [
    "EQNR.OL","TEL.OL","DNB.OL","MOWI.OL","NHY.OL","ORK.OL","RECSI.OL","SALM.OL",
    "SCHA.OL","STB.OL","SUBC.OL","TGS.OL","YAR.OL","AFG.OL","AKRBP.OL","AKSO.OL",
    "AMSC.OL","ATEA.OL","ATNO.OL","AUSS.OL","AVM.OL","AZEQ.OL","BAKKA.OL","BELCO.OL",
    "BEL.OL","BWLPG.OL","CADLR.OL","CECO.OL","CRAYN.OL","DLTX.OL","ECIT.OL","FLNG.OL",
    "FRO.OL","FUNCOM.OL","GJF.OL","GOGL.OL","GRIEG.OL","GSF.OL","HUNT.OL","HYON.OL",
    "JINHUI.OL","KOG.OL","KOMPLETT.OL","LSG.OL","MHG.OL","MNO.OL","NAL.OL","NEL.OL",
    "NEXT.OL","NFLX.OL","NORBT.OL","NOD.OL","NOM.OL","NONG.OL","NOR.OL","NRC.OL",
]

# ── BELGIUM ──────────────────────────────────────────────────
BEL20 = [
    "ABI.BR","UCB.BR","SOLB.BR","ACKB.BR","AGS.BR","APAM.BR","COLR.BR","GBL.BR",
    "ING.BR","PROX.BR","RAND.BR","TNET.BR","UMI.BR","ARGX.BR","BPOST.BR","COFB.BR",
    "EVS.BR","GLPG.BR","GIMB.BR","KBC.BR","MELX.BR","NXMB.BR","ONTEX.BR","SBMO.BR",
    "TINC.BR","THRB.BR","WDP.BR","XIOR.BR",
]

# ── PORTUGAL ─────────────────────────────────────────────────
PSI20 = [
    "EDP.LS","EDP-R.LS","EDPR.LS","GALP.LS","JMT.LS","BCP.LS","NOS.LS","SON.LS",
    "ALTR.LS","COR.LS","CTT.LS","EGL.LS","ESON.LS","GLOW.LS","GREENVOLT.LS","HPBV.LS",
    "IPGR.LS","MOTA.LS","NBA.LS","PHR.LS","RAM.LS","RAMO.LS","SEMAPA.LS","SEM.LS",
]

# ── IRELAND ──────────────────────────────────────────────────
ISEQ20 = [
    "AIBG.IR","AIB.IR","BIRG.IR","CRH.IR","DCC.IR","FBD.IR","GL9.IR","GFG.IR","ICG.IR",
    "INM.IR","IRES.IR","IWG.IR","KIN.IR","MGNI.IR","PLC.IR","SIGN.IR","SIG.IR","TPVG.IR",
    "THP.IR","UCG.IR",
]

# ── AUSTRIA ──────────────────────────────────────────────────
ATX = [
    "EBS.VI","OMV.VI","VIG.VI","ANDR.VI","ATS.VI","BAWAG.VI","CA.VI","EVN.VI","FACC.VI",
    "IIA.VI","IMMO.VI","LNZ.VI","MAYR.VI","NOEA.VI","POST.VI","RHI.VI","SANT.VI","SBO.VI",
    "STR.VI","TKA.VI","UQA.VI","VER.VI","WIE.VI","ZAG.VI",
]

# ── JAPAN ────────────────────────────────────────────────────

# Nikkei 225 — major constituents
NIKKEI225 = [
    "7203.T","6758.T","8306.T","9432.T","4502.T","6861.T","8058.T","7267.T","6902.T","4063.T",
    "9984.T","6501.T","7751.T","4661.T","8001.T","3382.T","8035.T","6702.T","8031.T","7974.T",
    "6594.T","9433.T","4519.T","8411.T","7832.T","6301.T","7011.T","9020.T","8802.T","8316.T",
    "4543.T","2914.T","6971.T","6503.T","7201.T","8309.T","8604.T","6762.T","7733.T","6752.T",
    "4507.T","5401.T","8001.T","8801.T","4568.T","9613.T","6098.T","8766.T","4151.T","7261.T",
    "2801.T","9503.T","7955.T","6981.T","1808.T","6473.T","8830.T","9022.T","2282.T","4452.T",
    "7912.T","5411.T","4704.T","4183.T","6326.T","7735.T","7741.T","9719.T","8002.T","6506.T",
    "1928.T","3407.T","4042.T","6367.T","8267.T","9101.T","8253.T","4005.T","9202.T","5201.T",
    "7270.T","2531.T","4755.T","6103.T","2002.T","4901.T","1332.T","3105.T","5802.T","9062.T",
    "9008.T","4307.T","4872.T","8233.T","8601.T","4217.T","5108.T","8750.T","9001.T","6645.T",
    "4021.T","7205.T","2768.T","5020.T","7272.T","6701.T","6841.T","7762.T","3289.T","6479.T",
    "6966.T","8905.T","4911.T","4324.T","9501.T","8331.T","7013.T","9602.T","4901.T","6361.T",
    "7211.T","1963.T","3099.T","4202.T","8725.T","7270.T","8252.T","3401.T","1801.T","6504.T",
    "5301.T","6954.T","4768.T","9983.T","4523.T","8830.T","6988.T","2502.T","1802.T","8028.T",
    "7731.T","6723.T","4208.T","4578.T","3092.T","6857.T","8803.T","6367.T","5019.T","2503.T",
    "4004.T","6952.T","1803.T","2501.T","3863.T","4188.T","9531.T","1925.T","6472.T","5202.T",
    "2413.T","4061.T","5332.T","7269.T","4911.T","4902.T","9064.T","8905.T","8233.T","6758.T",
]

# ── CHINA ────────────────────────────────────────────────────

# Hang Seng + ADRs
HANG_SENG = [
    "BABA","JD","PDD","BIDU","NIO","XPEV","LI","NTES","BILI","EDU",
    "TAL","VIPS","ZTO","RLX","BOSS","BZ","CAN","CNF","COVA","CRIS",
    "CRH","CSA","CSGP","DQ","DADA","DCI","DDL","DF","DG","DHC",
    "GOTU","GDS","GSX","HCM","HK","HTHT","IQ","KALA","KNSL","KRO",
    "LX","LKNCY","MCHI","MELI","MNSO","MPNGY","NVAX","OCA","OCSL","PICC",
    "SOS","TIGR","TMHC","TCOM","TRIP","TSM","TUYA","UCO","UTI","VNET",
    "WB","WUBA","XD","YMM","YSG","ZLAB","ZNH","ZTO",
    # H-shares and ADRs
    "TCEHY","700.HK","9988.HK","9618.HK","1810.HK","3690.HK","2318.HK",
    "939.HK","1299.HK","2628.HK","883.HK","2388.HK","1398.HK","3988.HK",
    "0011.HK","0002.HK","0003.HK","0006.HK","0012.HK","0016.HK","0017.HK",
    "0019.HK","0023.HK","0066.HK","0083.HK","0101.HK","0151.HK","0175.HK",
    "0267.HK","0288.HK","0291.HK","0386.HK","0388.HK","0669.HK","0688.HK",
    "0762.HK","0823.HK","0857.HK","0868.HK","0902.HK","0941.HK","0960.HK",
    "0968.HK","0981.HK","0992.HK","1038.HK","1044.HK","1109.HK","1177.HK",
    "1211.HK","1288.HK","1378.HK","1776.HK","1928.HK","1997.HK","2007.HK",
    "2269.HK","2313.HK","2319.HK","2331.HK","2333.HK","2382.HK","2588.HK",
    "2600.HK","2899.HK","3328.HK","3690.HK","3888.HK","3968.HK","3998.HK",
    "6098.HK","6160.HK","6862.HK","9618.HK","9888.HK","9999.HK",
]

# ── INDIA ────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS","ICICIBANK.NS",
    "SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS",
    "MARUTI.NS","BAJFINANCE.NS","HCLTECH.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS",
    "WIPRO.NS","ONGC.NS","POWERGRID.NS","NTPC.NS","COALINDIA.NS","BAJAJFINSV.NS",
    "ADANIGREEN.NS","ADANIPORTS.NS","ADANIENT.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "TECHM.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS","MM.NS","NESTLEIND.NS",
    "SBILIFE.NS","SHREECEM.NS","TATACONSUM.NS","TATAMOTORS.NS","TATAPOWER.NS",
    "BRITANNIA.NS","CIPLA.NS","APOLLOHOSP.NS","BAJAJ-AUTO.NS",
    # ADRs
    "INFY","WIT","HDB","IBN","ICICIB","MTE","HDFC","AXISB",
]

# ── SOUTH KOREA ──────────────────────────────────────────────
KOSPI_TOP = [
    "005930.KS","000660.KS","035420.KS","051910.KS","005380.KS","006400.KS","035720.KS",
    "068270.KS","105560.KS","028260.KS","207940.KS","012330.KS","009150.KS","000270.KS",
    "003550.KS","066570.KS","032830.KS","017670.KS","086790.KS","018260.KS","051900.KS",
    "011070.KS","024110.KS","033780.KS","096770.KS","010130.KS","011200.KS","012450.KS",
    "036460.KS","055550.KS","003670.KS","015760.KS","030200.KS","034730.KS","047050.KS",
    "064350.KS","078930.KS","090430.KS","097950.KS","139480.KS","180640.KS","214420.KS",
    "247540.KS","267250.KS","316140.KS","326030.KS","352820.KS","373220.KS","402340.KS",
]

# ── TAIWAN ───────────────────────────────────────────────────
TWSE_TOP = [
    "TSM","ASX","ASMPTF","HIMXF","UMC","CHIQ","ACMR","SIMO","CEVA","SPIL",
    "2330.TW","2317.TW","2454.TW","2308.TW","2303.TW","2412.TW","2882.TW","2891.TW",
    "2002.TW","1301.TW","2886.TW","2887.TW","1303.TW","2881.TW","2884.TW","2357.TW",
    "2885.TW","3711.TW","3034.TW","4938.TW","2395.TW","2379.TW","2344.TW","6505.TW",
    "2880.TW","2883.TW","2892.TW","1216.TW","2408.TW","2382.TW","2609.TW","2615.TW",
]

# ── SINGAPORE ────────────────────────────────────────────────
STI = [
    "D05.SI","O39.SI","U11.SI","Z74.SI","C6L.SI","S63.SI","F34.SI","V03.SI","G13.SI",
    "U96.SI","BN4.SI","C09.SI","C07.SI","C52.SI","D01.SI","E5H.SI","F99.SI","H78.SI",
    "J36.SI","J37.SI","K71.SI","ME8U.SI","N2IU.SI","Q01.SI","S58.SI","S68.SI","T39.SI",
    "U09.SI","U14.SI","V2C.SI","Y92.SI","T82U.SI","CJLU.SI","M44U.SI","KDCREIT.SI",
]

# ── AUSTRALIA ────────────────────────────────────────────────
ASX200_TOP = [
    "BHP.AX","CBA.AX","CSL.AX","ANZ.AX","NAB.AX","WBC.AX","MQG.AX","WES.AX","RIO.AX",
    "TLS.AX","FMG.AX","WOW.AX","GMG.AX","XRO.AX","RMD.AX","TCL.AX","ALL.AX","COL.AX",
    "SHL.AX","REA.AX","MP1.AX","AZJ.AX","WTC.AX","CPU.AX","STO.AX","NCM.AX","ORI.AX",
    "QBE.AX","SUN.AX","IAG.AX","APA.AX","DXS.AX","MIN.AX","GPT.AX","ASX.AX","BXB.AX",
    "CAR.AX","CHC.AX","CWN.AX","EVN.AX","HLS.AX","IEL.AX","JHX.AX","LLC.AX","MGR.AX",
    "NHF.AX","NST.AX","OZL.AX","PXA.AX","SCG.AX","SGR.AX","SFR.AX","SKI.AX","STW.AX",
    "SUL.AX","TWE.AX","VCX.AX","VVR.AX","WAF.AX","WGX.AX","WHC.AX","ZIM.AX",
]

# ── SAUDI ARABIA / GULF ──────────────────────────────────────
TADAWUL_TOP = [
    "2222.SR","1120.SR","2010.SR","1150.SR","1180.SR","2330.SR","2350.SR","3030.SR",
    "4200.SR","4210.SR","4160.SR","1020.SR","1050.SR","1060.SR","1080.SR","1211.SR",
    "2280.SR","2380.SR","4001.SR","4003.SR","4007.SR","4031.SR","4040.SR","4050.SR",
    "4051.SR","4061.SR","4100.SR","4110.SR","4130.SR","4140.SR","4162.SR","4180.SR",
    "4190.SR","4191.SR","6002.SR","6004.SR","6010.SR","6012.SR","6014.SR","6040.SR",
    "6090.SR","7010.SR","7020.SR","7030.SR","7040.SR","8010.SR","8020.SR","8030.SR",
    "8040.SR","8050.SR",
]

# UAE / Qatar / Kuwait (ADR format or international)
GULF_INTL = [
    "EMAAR","DIB","FAB","ADCB","TAQA.AE","ALDAR","DEWA","FERTIGLOBE",
    "QNBK.QA","MARK.QA","CBQK.QA","GISS.QA","IQCD.QA","BRES.QA",
    "NBK.KW","CBK.KW","BURGAN.KW","ZAIN.KW","KIB.KW",
]

# ── RUSSIA / EASTERN EUROPE ──────────────────────────────────
# Note: Russian stocks have very limited liquidity since 2022.
# Using mainly ADRs / London-listed that still trade internationally.
RUSSIA_INTL = [
    "LUKOY","NVTK","ROSN","TATNL","SBER","GAZP","YNDX","FIVE","MGNT",
    "AFKS.ME","ALRS.ME","CHMF.ME","GMKN.ME","HYDR.ME","IRAO.ME","LKOH.ME",
    "MAGN.ME","MOEX.ME","MTSS.ME","NLMK.ME","NVTK.ME","PHOR.ME","PIKK.ME",
    "PLZL.ME","POLY.ME","ROSN.ME","RTKM.ME","SBER.ME","SNGS.ME","TATN.ME",
    "TCSG.ME","TRNFP.ME","UPRO.ME","VEON","OZON","FIXP",
]

# Poland, Czech, Hungary
CEE_INTL = [
    "PKN.WA","PKO.WA","PZU.WA","CDR.WA","LPP.WA","ALE.WA","CCC.WA","DNP.WA",
    "JSW.WA","KGH.WA","KRU.WA","MBK.WA","MRC.WA","OPL.WA","PEO.WA","PLY.WA",
    "SPL.WA","TPE.WA","VRG.WA",
    "CEZ.PR","KOMB.PR","O2.PR","EINE.PR","RELN.PR","RBAG.PR",
    "MOL.BD","OTP.BD","MTELEKOM.BD","RICHTER.BD","OPUS.BD",
]

# ── EMERGING MARKETS ─────────────────────────────────────────

# South Africa — JSE Top 40
JSE_TOP = [
    "AGL.JO","BIL.JO","NPN.JO","SOL.JO","FSR.JO","SBK.JO","MTN.JO","VOD.JO","MNP.JO",
    "ABG.JO","AMS.JO","BVT.JO","CFR.JO","CPI.JO","DSY.JO","GRT.JO","HAR.JO","INL.JO",
    "INP.JO","KIO.JO","LHC.JO","MRP.JO","MSM.JO","NED.JO","OML.JO","RDF.JO","REM.JO",
    "RNI.JO","SHP.JO","SLM.JO","SPP.JO","TBS.JO","TFG.JO","TRU.JO","WHL.JO",
]

# Turkey — BIST 30
BIST30 = [
    "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS","ENKAI.IS","EREGL.IS",
    "FROTO.IS","GARAN.IS","GUBRF.IS","HALKB.IS","ISCTR.IS","KCHOL.IS","KOZAA.IS",
    "KOZAL.IS","KRDMD.IS","MGROS.IS","ODAS.IS","PETKM.IS","PGSUS.IS","SAHOL.IS",
    "SASA.IS","SISE.IS","SOKM.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS",
    "TOASO.IS","TTKOM.IS","TUPRS.IS","VAKBN.IS","YKBNK.IS",
]

# Indonesia — IDX top 30
IDX_TOP = [
    "BBCA.JK","BBRI.JK","BMRI.JK","TLKM.JK","ASII.JK","BBNI.JK","UNVR.JK","HMSP.JK",
    "ICBP.JK","KLBF.JK","GGRM.JK","SMGR.JK","PTBA.JK","INDF.JK","JSMR.JK","PWON.JK",
    "PGAS.JK","MNCN.JK","WIKA.JK","ADHI.JK","UNTR.JK","SCMA.JK","PTPP.JK","ANTM.JK",
]

# ── MASTER UNIVERSE DICT ─────────────────────────────────────

UNIVERSE = {
    "Americas": {
        "🇺🇸 S&P 500": SP500,
        "🇺🇸 S&P 400 (Mid Cap)": SP400,
        "🇺🇸 S&P 600 (Small Cap)": SP600,
        "🇨🇦 TSX 60 (Canada)": TSX60,
        "🇧🇷 Bovespa (Brazil)": BOVESPA,
        "🇲🇽 IPC (Mexico)": IPC_MEXICO,
    },
    "France": {
        "🇫🇷 CAC 40": CAC40,
        "🇫🇷 SBF 120 (Mid Cap)": SBF120_EXTRA,
    },
    "Germany": {
        "🇩🇪 DAX 40": DAX40,
        "🇩🇪 MDAX (Mid Cap)": MDAX_EXTRA,
    },
    "United Kingdom": {
        "🇬🇧 FTSE 100": FTSE100,
        "🇬🇧 FTSE 250 (Mid Cap)": FTSE250_EXTRA,
    },
    "Spain": {
        "🇪🇸 IBEX 35": IBEX35,
    },
    "Italy": {
        "🇮🇹 FTSE MIB": FTSE_MIB,
    },
    "Europe (Full)": {
        "🇳🇱 AEX (Netherlands)": AEX,
        "🇸🇪 OMX Stockholm 30": OMXS30,
        "🇩🇰 OMX Copenhagen 25": OMXC25,
        "🇨🇭 SMI (Switzerland)": SMI,
        "🇫🇮 OMX Helsinki 25": OMXH25,
        "🇳🇴 OBX (Norway)": OBX,
        "🇧🇪 BEL 20 (Belgium)": BEL20,
        "🇵🇹 PSI 20 (Portugal)": PSI20,
        "🇮🇪 ISEQ 20 (Ireland)": ISEQ20,
        "🇦🇹 ATX (Austria)": ATX,
    },
    "Asia & Middle East": {
        "🇯🇵 Nikkei 225 (Japan)": NIKKEI225,
        "🇨🇳 Hang Seng / ADR (China)": HANG_SENG,
        "🇮🇳 Nifty 50 / ADR (India)": NIFTY50,
        "🇰🇷 KOSPI (South Korea)": KOSPI_TOP,
        "🇹🇼 TWSE (Taiwan)": TWSE_TOP,
        "🇸🇬 STI (Singapore)": STI,
        "🇦🇺 ASX 200 (Australia)": ASX200_TOP,
        "🇸🇦 Tadawul (Saudi Arabia)": TADAWUL_TOP,
        "🌍 Gulf / UAE / Qatar": GULF_INTL,
        "🇷🇺 Russia / Eastern Europe": RUSSIA_INTL + CEE_INTL,
    },
}

ALL_REGIONS = list(UNIVERSE.keys())

def get_tickers_for_regions(regions: list) -> list:
    tickers = []
    for region in regions:
        for subgroup, tkrs in UNIVERSE.get(region, {}).items():
            tickers.extend(tkrs)
    return list(dict.fromkeys(tickers))  # deduplicate preserving order

def count_universe():
    total = 0
    for region, groups in UNIVERSE.items():
        r_total = sum(len(v) for v in groups.values())
        print(f"  {region}: {r_total}")
        total += r_total
    print(f"  TOTAL: {total}")

if __name__ == "__main__":
    count_universe()
