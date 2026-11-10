def page_anomaly_report(pages):
    out=[]
    for p in pages:
        n=len(p["text"])
        flags=[]
        if n<80: flags.append("Very little extracted text")
        if p.get("ocr_used"): flags.append("OCR fallback used — verify manually")
        if not p.get("sections"): flags.append("No numbered section heading detected")
        if n>12000: flags.append("Long page — inspect reading order / chunking")
        out.append({"Document":p["document"],"Page":p["page"],"Characters":n,
                    "OCR":p.get("ocr_used",False),"Tables":len(p.get("tables",[])),
                    "Diagnostics":"; ".join(flags) if flags else "No basic anomaly detected"})
    return out
