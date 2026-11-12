import io, re
import fitz
import pdfplumber

SECTION_RE=re.compile(r'(?im)^\s*((?:section|article|clause|§)\s*)?(\d+(?:\.\d+){0,5}[A-Za-z]?)\s*[\.)\-:]?\s+(.{2,120})?$')

def _ocr_page(page):
    try:
        import pytesseract
        from PIL import Image
        pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
        img=Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 3")
    except Exception:
        return None

def ingest_pdf(pdf_bytes, filename):
    doc=fitz.open(stream=pdf_bytes,filetype="pdf")
    out=[]
    try:
        plumber=pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception: plumber=None
    for i,page in enumerate(doc):
        blocks=page.get_text("blocks",sort=True)
        # Reading-order sort: top-to-bottom then left-to-right; retain text blocks and bbox.
        blocks=sorted([b for b in blocks if len(b)>4 and str(b[4]).strip()],key=lambda b:(round(b[1]/5),b[0]))
        text="\n".join(str(b[4]).strip() for b in blocks)
        ocr_used=False
        if len(text.strip())<35:
            od=_ocr_page(page)
            if od:
                words=[]
                for j,w in enumerate(od["text"]):
                    if w.strip() and int(od["conf"][j])>=0:
                        words.append((int(od["top"][j]),int(od["left"][j]),w))
                text=" ".join(w[2] for w in sorted(words))
                ocr_used=bool(text.strip())
        tables=[]
        if plumber and i<len(plumber.pages):
            try:
                tables=[t for t in (plumber.pages[i].extract_tables() or []) if t]
            except Exception: pass
        sections=[]
        for line in text.splitlines():
            m=SECTION_RE.match(line.strip())
            if m:
                sections.append((m.group(2)+" "+(m.group(3) or "")).strip()[:120])
        # Include tables as searchable text without losing structured representation.
        table_text=[]
        for ti,t in enumerate(tables):
            table_text.append(f"TABLE {ti+1}\n"+"\n".join(" | ".join(str(c or "") for c in row) for row in t))
        full=text+("\n\n"+"\n\n".join(table_text) if table_text else "")
        # Page-level chunk; preserve source bbox list to allow traceable region citations.
        bbox=[list(map(float,b[:4])) for b in blocks]
        out.append({"document":filename,"page":i+1,"text":full,"raw_text":text,"tables":tables,
                    "sections":sections,"bbox":bbox,"ocr_used":ocr_used})
    if plumber: plumber.close()
    doc.close()
    return out
