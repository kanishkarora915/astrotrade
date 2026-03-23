"""
AstroNifty — Daily PDF Report Exporter
Generates a comprehensive 12-page daily PDF report with ALL data from the trading day.
Runs at 4:00 PM IST after market close.

AstroTrade by Kanishk Arora
"""

import io
import os
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Rect
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════
# COLOR PALETTE — Professional dark-accented theme on white
# ═══════════════════════════════════════════════════════════════════════
class Theme:
    """Clean white theme with colored accents."""
    # Core
    BG_WHITE = colors.white
    BG_DARK = colors.HexColor("#1A1A2E")
    BG_COVER = colors.HexColor("#0F0F23")
    BG_HEADER = colors.HexColor("#16213E")
    BG_SECTION = colors.HexColor("#F7F8FA")

    # Text
    TEXT_DARK = colors.HexColor("#1A1A2E")
    TEXT_LIGHT = colors.white
    TEXT_MUTED = colors.HexColor("#6C757D")
    TEXT_ACCENT = colors.HexColor("#00E5FF")

    # Status
    GREEN = colors.HexColor("#00C853")
    RED = colors.HexColor("#FF1744")
    AMBER = colors.HexColor("#FFD600")
    BLUE = colors.HexColor("#2979FF")

    # Table
    TABLE_HEADER_BG = colors.HexColor("#1A1A2E")
    TABLE_HEADER_FG = colors.white
    TABLE_ROW_EVEN = colors.HexColor("#F7F8FA")
    TABLE_ROW_ODD = colors.white
    TABLE_BORDER = colors.HexColor("#DEE2E6")

    # Charts
    CHART_BG = "#FFFFFF"
    CHART_CE_COLOR = "#FF1744"
    CHART_PE_COLOR = "#00C853"
    CHART_LINE_COLOR = "#2979FF"
    CHART_GRID_COLOR = "#E0E0E0"
    CHART_FII_COLOR = "#2979FF"
    CHART_DII_COLOR = "#FFD600"


# ═══════════════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════════════
def _build_styles() -> dict:
    """Build all paragraph styles for the report."""
    ss = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "CoverTitle", parent=ss["Title"],
        fontName="Helvetica-Bold", fontSize=32, leading=38,
        textColor=Theme.TEXT_LIGHT, alignment=TA_CENTER,
        spaceAfter=10,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "CoverSubtitle", parent=ss["Normal"],
        fontName="Helvetica", fontSize=14, leading=18,
        textColor=Theme.TEXT_ACCENT, alignment=TA_CENTER,
        spaceAfter=6,
    )
    styles["cover_info"] = ParagraphStyle(
        "CoverInfo", parent=ss["Normal"],
        fontName="Helvetica", fontSize=11, leading=14,
        textColor=colors.HexColor("#AAAACC"), alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["page_title"] = ParagraphStyle(
        "PageTitle", parent=ss["Heading1"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=Theme.BG_DARK, alignment=TA_LEFT,
        spaceAfter=12, spaceBefore=4,
        borderWidth=0, borderColor=Theme.TEXT_ACCENT,
        borderPadding=0,
    )
    styles["section_title"] = ParagraphStyle(
        "SectionTitle", parent=ss["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=Theme.BLUE, alignment=TA_LEFT,
        spaceAfter=6, spaceBefore=10,
    )
    styles["body"] = ParagraphStyle(
        "Body", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=Theme.TEXT_DARK, alignment=TA_LEFT,
        spaceAfter=4,
    )
    styles["body_bold"] = ParagraphStyle(
        "BodyBold", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=Theme.TEXT_DARK, alignment=TA_LEFT,
        spaceAfter=4,
    )
    styles["value_green"] = ParagraphStyle(
        "ValueGreen", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=Theme.GREEN, alignment=TA_LEFT,
    )
    styles["value_red"] = ParagraphStyle(
        "ValueRed", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=Theme.RED, alignment=TA_LEFT,
    )
    styles["footer"] = ParagraphStyle(
        "Footer", parent=ss["Normal"],
        fontName="Helvetica", fontSize=7, leading=9,
        textColor=Theme.TEXT_MUTED, alignment=TA_CENTER,
    )
    styles["kv_label"] = ParagraphStyle(
        "KVLabel", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=Theme.TEXT_MUTED, alignment=TA_LEFT,
    )
    styles["kv_value"] = ParagraphStyle(
        "KVValue", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=Theme.TEXT_DARK, alignment=TA_LEFT,
    )
    styles["small_note"] = ParagraphStyle(
        "SmallNote", parent=ss["Normal"],
        fontName="Helvetica-Oblique", fontSize=7, leading=9,
        textColor=Theme.TEXT_MUTED, alignment=TA_LEFT,
    )
    return styles


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _safe_get(data: Any, key: str, default: Any = "—") -> Any:
    """Safely get a value from a dict, list-of-dicts, or object."""
    if data is None:
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0].get(key, default)
    if hasattr(data, key):
        return getattr(data, key, default)
    return default


def _fmt_num(val: Any, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    """Format a number with commas and decimals."""
    if val is None or val == "—":
        return "—"
    try:
        num = float(val)
        if decimals == 0:
            formatted = f"{int(num):,}"
        else:
            formatted = f"{num:,.{decimals}f}"
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _pnl_color(val: Any) -> colors.Color:
    """Return green for positive, red for negative P&L."""
    try:
        return Theme.GREEN if float(val) >= 0 else Theme.RED
    except (ValueError, TypeError):
        return Theme.TEXT_DARK


def _pnl_str(val: Any, prefix: str = "") -> str:
    """Format P&L with sign."""
    try:
        num = float(val)
        sign = "+" if num >= 0 else ""
        return f"{prefix}{sign}{num:,.2f}"
    except (ValueError, TypeError):
        return "—"


def _degree_to_sign(deg: Any) -> str:
    """Convert degree to zodiac sign."""
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    ]
    try:
        d = float(deg) % 360
        idx = int(d // 30)
        return signs[idx]
    except (ValueError, TypeError):
        return "—"


def _make_styled_table(
    data: List[list],
    col_widths: Optional[List[float]] = None,
    has_header: bool = True,
) -> Table:
    """Create a professionally styled table."""
    if not data:
        return Table([["No data available"]])

    table = Table(data, colWidths=col_widths, repeatRows=1 if has_header else 0)

    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, Theme.TABLE_BORDER),
    ]

    if has_header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), Theme.TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), Theme.TABLE_HEADER_FG),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
        ]

    # Alternating row colors
    for i in range(1, len(data)):
        bg = Theme.TABLE_ROW_EVEN if i % 2 == 0 else Theme.TABLE_ROW_ODD
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

    table.setStyle(TableStyle(style_cmds))
    return table


def _chart_to_image(fig: plt.Figure, width: float = 16 * cm, height: float = 8 * cm) -> Image:
    """Convert a matplotlib figure to a ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=Theme.CHART_BG)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)


# ═══════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def _create_pcr_chart(oi_data: dict) -> Optional[Image]:
    """Create OI distribution bar chart for CE vs PE OI across strikes."""
    chain = oi_data.get("chain_json") or oi_data.get("chain") or []
    if isinstance(chain, dict):
        chain = chain.get("strikes", chain.get("data", []))
    if not chain or not isinstance(chain, list):
        return None

    try:
        strikes = []
        ce_oi_vals = []
        pe_oi_vals = []
        for row in chain:
            if isinstance(row, dict):
                s = row.get("strike") or row.get("strike_price")
                ce = row.get("ce_oi", 0) or 0
                pe = row.get("pe_oi", 0) or 0
                if s is not None:
                    strikes.append(float(s))
                    ce_oi_vals.append(float(ce))
                    pe_oi_vals.append(float(pe))

        if not strikes:
            return None

        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor(Theme.CHART_BG)
        ax.set_facecolor(Theme.CHART_BG)

        x = np.arange(len(strikes))
        bar_w = 0.38
        ax.barh(x - bar_w / 2, [-v / 1e5 for v in ce_oi_vals], bar_w,
                label="CE OI (Lakhs)", color=Theme.CHART_CE_COLOR, alpha=0.85)
        ax.barh(x + bar_w / 2, [v / 1e5 for v in pe_oi_vals], bar_w,
                label="PE OI (Lakhs)", color=Theme.CHART_PE_COLOR, alpha=0.85)

        ax.set_yticks(x[::max(1, len(x) // 15)])
        ax.set_yticklabels([f"{int(s)}" for s in strikes[::max(1, len(x) // 15)]],
                           fontsize=7)
        ax.set_xlabel("OI (Lakhs)", fontsize=8, color="#333")
        ax.set_title("OI Distribution — CE vs PE", fontsize=10, fontweight="bold", color="#1A1A2E")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3, color=Theme.CHART_GRID_COLOR)
        ax.axvline(x=0, color="#333", linewidth=0.5)
        plt.tight_layout()
        return _chart_to_image(fig, width=17 * cm, height=9 * cm)
    except Exception as e:
        logger.warning("Failed to create PCR chart: {}", str(e))
        return None


def _create_iv_curve_chart(greeks_data: dict) -> Optional[Image]:
    """Create IV curve across strikes."""
    iv_skew = greeks_data.get("iv_skew") or greeks_data.get("iv_data") or []
    if not iv_skew or not isinstance(iv_skew, list):
        return None

    try:
        strikes = []
        ce_ivs = []
        pe_ivs = []
        for row in iv_skew:
            if isinstance(row, dict):
                s = row.get("strike") or row.get("strike_price")
                ce_iv = row.get("ce_iv") or row.get("call_iv", 0)
                pe_iv = row.get("pe_iv") or row.get("put_iv", 0)
                if s is not None:
                    strikes.append(float(s))
                    ce_ivs.append(float(ce_iv or 0))
                    pe_ivs.append(float(pe_iv or 0))

        if not strikes:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(Theme.CHART_BG)
        ax.set_facecolor(Theme.CHART_BG)

        ax.plot(strikes, ce_ivs, color=Theme.CHART_CE_COLOR, linewidth=1.8,
                marker="o", markersize=3, label="CE IV", alpha=0.9)
        ax.plot(strikes, pe_ivs, color=Theme.CHART_PE_COLOR, linewidth=1.8,
                marker="o", markersize=3, label="PE IV", alpha=0.9)

        ax.set_xlabel("Strike", fontsize=8)
        ax.set_ylabel("IV (%)", fontsize=8)
        ax.set_title("Implied Volatility Curve", fontsize=10, fontweight="bold", color="#1A1A2E")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, color=Theme.CHART_GRID_COLOR)
        plt.tight_layout()
        return _chart_to_image(fig, width=17 * cm, height=7 * cm)
    except Exception as e:
        logger.warning("Failed to create IV curve chart: {}", str(e))
        return None


def _create_fii_dii_chart(fii_dii_data: dict) -> Optional[Image]:
    """Create FII/DII net flow bar chart."""
    try:
        fii_net = float(fii_dii_data.get("fii_net", 0) or 0)
        dii_net = float(fii_dii_data.get("dii_net", 0) or 0)

        if fii_net == 0 and dii_net == 0:
            return None

        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor(Theme.CHART_BG)
        ax.set_facecolor(Theme.CHART_BG)

        categories = ["FII Net", "DII Net", "Total Net"]
        values = [fii_net, dii_net, fii_net + dii_net]
        bar_colors = [
            Theme.CHART_FII_COLOR if fii_net >= 0 else Theme.CHART_CE_COLOR,
            Theme.CHART_DII_COLOR if dii_net >= 0 else Theme.CHART_CE_COLOR,
            Theme.CHART_PE_COLOR if (fii_net + dii_net) >= 0 else Theme.CHART_CE_COLOR,
        ]

        bars = ax.bar(categories, [v / 100 for v in values], color=bar_colors, alpha=0.85, width=0.5)

        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:,.0f} Cr", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold", color="#333")

        ax.set_ylabel("Amount (x100 Cr)", fontsize=8)
        ax.set_title("FII / DII Net Cash Flow", fontsize=10, fontweight="bold", color="#1A1A2E")
        ax.axhline(y=0, color="#333", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3, color=Theme.CHART_GRID_COLOR)
        plt.tight_layout()
        return _chart_to_image(fig, width=14 * cm, height=6 * cm)
    except Exception as e:
        logger.warning("Failed to create FII/DII chart: {}", str(e))
        return None


def _create_sector_heatmap(sector_data: dict) -> Optional[Image]:
    """Create sector performance heatmap-style horizontal bar chart."""
    sectors = sector_data if isinstance(sector_data, dict) else {}
    if not sectors:
        return None

    try:
        names = []
        changes = []
        for name, info in sectors.items():
            if isinstance(info, dict):
                chg = info.get("change_pct", 0) or 0
                names.append(str(name))
                changes.append(float(chg))
            elif isinstance(info, (int, float)):
                names.append(str(name))
                changes.append(float(info))

        if not names:
            return None

        # Sort by change
        paired = sorted(zip(names, changes), key=lambda x: x[1])
        names, changes = zip(*paired)

        fig, ax = plt.subplots(figsize=(10, max(3.5, len(names) * 0.4)))
        fig.patch.set_facecolor(Theme.CHART_BG)
        ax.set_facecolor(Theme.CHART_BG)

        bar_colors = [Theme.CHART_PE_COLOR if c >= 0 else Theme.CHART_CE_COLOR for c in changes]
        bars = ax.barh(names, changes, color=bar_colors, alpha=0.85, height=0.6)

        for bar, val in zip(bars, changes):
            x_pos = bar.get_width()
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f" {val:+.2f}%", ha="left" if val >= 0 else "right",
                    va="center", fontsize=7, fontweight="bold", color="#333")

        ax.set_xlabel("Change %", fontsize=8)
        ax.set_title("Sector Performance", fontsize=10, fontweight="bold", color="#1A1A2E")
        ax.axvline(x=0, color="#333", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3, color=Theme.CHART_GRID_COLOR)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        return _chart_to_image(fig, width=17 * cm, height=max(6, len(names) * 0.7) * cm)
    except Exception as e:
        logger.warning("Failed to create sector heatmap: {}", str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN PDF EXPORTER CLASS
# ═══════════════════════════════════════════════════════════════════════

class DailyPDFExporter:
    """
    Generates a comprehensive daily PDF report with ALL data from the trading day.
    Runs at 4:00 PM IST after market close.
    """

    def __init__(self, exports_dir: Optional[str] = None):
        self.exports_dir = Path(exports_dir) if exports_dir else Path(__file__).parent.parent.parent / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.styles = _build_styles()
        self.generated_at = datetime.now()
        logger.info("DailyPDFExporter initialized. Exports dir: {}", self.exports_dir)

    # ──────────────────────────────────────────────────────────────
    # Footer / Header drawing
    # ──────────────────────────────────────────────────────────────

    def _draw_footer(self, canvas, doc):
        """Draw footer on every page."""
        canvas.saveState()
        page_w, page_h = A4

        # Footer line
        canvas.setStrokeColor(Theme.TABLE_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(1.5 * cm, 1.2 * cm, page_w - 1.5 * cm, 1.2 * cm)

        # Footer text
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(Theme.TEXT_MUTED)

        footer_left = "ASTRONIFTY ENGINE \u2014 AstroTrade by Kanishk Arora"
        footer_right = f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M IST')}"
        page_num = f"Page {doc.page}"

        canvas.drawString(1.5 * cm, 0.8 * cm, footer_left)
        canvas.drawRightString(page_w - 1.5 * cm, 0.8 * cm, footer_right)
        canvas.drawCentredString(page_w / 2, 0.8 * cm, page_num)
        canvas.restoreState()

    def _draw_cover_bg(self, canvas, doc):
        """Draw dark cover page background."""
        canvas.saveState()
        page_w, page_h = A4
        canvas.setFillColor(Theme.BG_COVER)
        canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        # Accent line at top
        canvas.setFillColor(Theme.TEXT_ACCENT)
        canvas.rect(0, page_h - 4 * mm, page_w, 4 * mm, fill=1, stroke=0)

        # Accent line at bottom
        canvas.rect(0, 0, page_w, 4 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def _draw_page_header(self, canvas, doc):
        """Draw header and footer on content pages."""
        canvas.saveState()
        page_w, page_h = A4

        # Top accent bar
        canvas.setFillColor(Theme.BG_DARK)
        canvas.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=1, stroke=0)

        canvas.setFillColor(Theme.TEXT_ACCENT)
        canvas.rect(0, page_h - 8.5 * mm, page_w, 0.5 * mm, fill=1, stroke=0)

        # Header text
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(Theme.TEXT_LIGHT)
        canvas.drawString(1.5 * cm, page_h - 6 * mm, "ASTRONIFTY DAILY REPORT")
        canvas.drawRightString(page_w - 1.5 * cm, page_h - 6 * mm,
                               self.report_date_str)

        canvas.restoreState()
        self._draw_footer(canvas, doc)

    # ──────────────────────────────────────────────────────────────
    # Main generation entry point
    # ──────────────────────────────────────────────────────────────

    def generate_daily_report(
        self,
        report_date: Optional[date] = None,
        db_manager=None,
        engine_data: Optional[dict] = None,
    ) -> str:
        """
        Generate complete daily PDF report.

        Args:
            report_date: Date for the report (defaults to today).
            db_manager: Database manager with query methods.
            engine_data: Dict with all engine data for the day:
                scores, oi_data, astro_data, greeks_data, fii_dii_data,
                global_cues, sector_data, signals, trades, price_action,
                weekly_forecast, risk_summary, market_summary
        Returns:
            Absolute path to generated PDF file.
        """
        self.generated_at = datetime.now()
        report_date = report_date or date.today()
        self.report_date = report_date
        self.report_date_str = report_date.strftime("%A, %d %B %Y")

        data = engine_data or {}
        if not data and db_manager:
            data = self._fetch_from_db(db_manager, report_date)

        # Prepare output path
        date_str = report_date.strftime("%Y%m%d")
        day_dir = self.exports_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = day_dir / f"ASTRONIFTY_DAILY_{date_str}.pdf"

        logger.info("Generating PDF report for {} -> {}", date_str, pdf_path)

        # Build the PDF
        doc = BaseDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        page_w, page_h = A4
        frame_w = page_w - 3 * cm
        frame_h = page_h - 4 * cm

        cover_frame = Frame(
            1.5 * cm, 2 * cm, frame_w, frame_h,
            id="cover_frame",
        )
        content_frame = Frame(
            1.5 * cm, 2 * cm, frame_w, page_h - 4.5 * cm,
            id="content_frame",
        )

        cover_template = PageTemplate(
            id="cover", frames=[cover_frame],
            onPage=self._draw_cover_bg,
        )
        content_template = PageTemplate(
            id="content", frames=[content_frame],
            onPage=self._draw_page_header,
        )

        doc.addPageTemplates([cover_template, content_template])

        # Build story (all flowables)
        story = []
        story.extend(self._build_cover_page(data))
        story.append(NextPageTemplate("content"))
        story.append(PageBreak())
        story.extend(self._build_score_summary(data))
        story.append(PageBreak())
        story.extend(self._build_oi_chain_analysis(data))
        story.append(PageBreak())
        story.extend(self._build_astro_data(data))
        story.append(PageBreak())
        story.extend(self._build_greeks_iv(data))
        story.append(PageBreak())
        story.extend(self._build_fii_dii(data))
        story.append(PageBreak())
        story.extend(self._build_global_cues(data))
        story.append(PageBreak())
        story.extend(self._build_sector_analysis(data))
        story.append(PageBreak())
        story.extend(self._build_trade_log(data))
        story.append(PageBreak())
        story.extend(self._build_price_action(data))
        story.append(PageBreak())
        story.extend(self._build_weekly_probability(data))
        story.append(PageBreak())
        story.extend(self._build_risk_summary(data))

        doc.build(story)
        logger.info("PDF report generated: {} ({:.1f} KB)",
                     pdf_path, pdf_path.stat().st_size / 1024)
        return str(pdf_path)

    # ──────────────────────────────────────────────────────────────
    # Fetch data from database if no engine_data provided
    # ──────────────────────────────────────────────────────────────

    def _fetch_from_db(self, db_manager, report_date: date) -> dict:
        """Fetch all data from database for a given date."""
        data = {}
        fetch_map = {
            "oi_data": "get_todays_oi_snapshots",
            "astro_data": "get_todays_astro",
            "signals": "get_todays_signals",
            "fii_dii_data": "get_todays_fii_dii",
            "weekly_forecast": "get_weekly_forecast",
            "trades": "get_todays_trades",
            "pnl_summary": "get_todays_pnl",
        }
        for key, method_name in fetch_map.items():
            try:
                method = getattr(db_manager, method_name, None)
                if method:
                    result = method()
                    data[key] = result if result else {}
                else:
                    data[key] = {}
            except Exception as e:
                logger.warning("Failed to fetch {} from DB: {}", key, str(e))
                data[key] = {}

        # Try additional methods
        for key, method_name in [
            ("scores", "get_todays_scores"),
            ("greeks_data", "get_todays_greeks"),
            ("global_cues", "get_global_cues"),
            ("sector_data", "get_todays_sectors"),
            ("price_action", "get_price_action"),
            ("risk_summary", "get_risk_summary"),
            ("market_summary", "get_market_summary"),
        ]:
            try:
                method = getattr(db_manager, method_name, None)
                if method:
                    data[key] = method() or {}
                else:
                    data[key] = {}
            except Exception:
                data[key] = {}

        return data

    # ══════════════════════════════════════════════════════════════
    # PAGE 1 — COVER PAGE
    # ══════════════════════════════════════════════════════════════

    def _build_cover_page(self, data: dict) -> list:
        """Build cover page flowables."""
        elements = []
        s = self.styles

        elements.append(Spacer(1, 4 * cm))

        elements.append(Paragraph("ASTRONIFTY", s["cover_title"]))
        elements.append(Paragraph("DAILY REPORT", s["cover_title"]))
        elements.append(Spacer(1, 0.8 * cm))

        elements.append(Paragraph("AstroTrade by Kanishk Arora", s["cover_subtitle"]))
        elements.append(Spacer(1, 1 * cm))

        elements.append(Paragraph(self.report_date_str, s["cover_info"]))
        elements.append(Spacer(1, 2 * cm))

        # Market summary on cover
        market = data.get("market_summary", {})
        if not isinstance(market, dict):
            market = {}

        nifty_close = _safe_get(market, "nifty_close")
        nifty_chg = _safe_get(market, "nifty_change_pct")
        bn_close = _safe_get(market, "banknifty_close")
        bn_chg = _safe_get(market, "banknifty_change_pct")

        pnl_summary = data.get("pnl_summary", data.get("risk_summary", {}))
        if isinstance(pnl_summary, list) and len(pnl_summary) > 0:
            pnl_summary = pnl_summary[0] if isinstance(pnl_summary[0], dict) else {}
        if not isinstance(pnl_summary, dict):
            pnl_summary = {}
        daily_pnl = pnl_summary.get("daily_pnl", pnl_summary.get("total_pnl", "—"))

        trades = data.get("trades", [])
        if not isinstance(trades, list):
            trades = [trades] if trades else []
        total_trades = len(trades)
        winning = sum(1 for t in trades if isinstance(t, dict) and float(t.get("pnl", 0) or 0) > 0)
        win_rate = f"{(winning / total_trades * 100):.0f}%" if total_trades > 0 else "—"

        cover_data = [
            ["MARKET SUMMARY", "", "", ""],
            ["", "Close", "Change %", ""],
            ["NIFTY 50", _fmt_num(nifty_close), _fmt_num(nifty_chg, suffix="%"), ""],
            ["BANKNIFTY", _fmt_num(bn_close), _fmt_num(bn_chg, suffix="%"), ""],
            ["", "", "", ""],
            ["Day P&L", _pnl_str(daily_pnl), "Trades", str(total_trades)],
            ["Win Rate", win_rate, "", ""],
        ]

        cover_table = Table(cover_data, colWidths=[4.5 * cm, 4 * cm, 3.5 * cm, 3 * cm])
        cover_style = TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#CCCCDD")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            # Title row
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 13),
            ("TEXTCOLOR", (0, 0), (-1, 0), Theme.TEXT_ACCENT),
            ("SPAN", (0, 0), (-1, 0)),
            # Header row
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, 1), 8),
            ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#888899")),
            # Subtle border
            ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.HexColor("#333355")),
            ("LINEBELOW", (0, 3), (-1, 3), 0.5, colors.HexColor("#333355")),
        ])
        cover_table.setStyle(cover_style)
        elements.append(cover_table)

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 2 — SCORE SUMMARY
    # ══════════════════════════════════════════════════════════════

    def _build_score_summary(self, data: dict) -> list:
        """Build score summary page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("SCORE SUMMARY", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        scores = data.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}

        # Final scores table for each index
        elements.append(Paragraph("Final Composite Scores", s["section_title"]))

        index_scores_data = [["Index", "Total Score", "Bias", "Confidence", "Signal"]]
        for idx_name in ["NIFTY", "BANKNIFTY", "GIFTNIFTY"]:
            idx_score = scores.get(idx_name, scores.get(idx_name.lower(), {}))
            if not isinstance(idx_score, dict):
                idx_score = {}
            total = _safe_get(idx_score, "total_score", _safe_get(idx_score, "total", "—"))
            bias = _safe_get(idx_score, "bias", _safe_get(idx_score, "market_bias", "—"))
            conf = _safe_get(idx_score, "confidence", "—")
            signal = _safe_get(idx_score, "signal", _safe_get(idx_score, "signal_type", "—"))
            index_scores_data.append([idx_name, _fmt_num(total), str(bias), _fmt_num(conf), str(signal)])

        elements.append(_make_styled_table(
            index_scores_data,
            col_widths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm],
        ))
        elements.append(Spacer(1, 0.6 * cm))

        # Score breakdown (10 components)
        elements.append(Paragraph("Score Component Breakdown", s["section_title"]))

        from config import SCORING_WEIGHTS
        component_names = {
            "oi_chain": "OI Chain",
            "oi_buildup": "OI Buildup",
            "astro": "Astro",
            "greeks": "Greeks & IV",
            "price_action": "Price Action",
            "fii_dii": "FII / DII",
            "global_cues": "Global Cues",
            "smart_money": "Smart Money",
            "expiry": "Expiry Effect",
            "breadth": "Market Breadth",
        }

        breakdown_header = ["Component", "Max Weight"]
        idx_names_in_scores = ["NIFTY", "BANKNIFTY", "GIFTNIFTY"]
        for idx_name in idx_names_in_scores:
            breakdown_header.append(f"{idx_name} Score")

        breakdown_data = [breakdown_header]
        for comp_key, max_w in SCORING_WEIGHTS.items():
            row = [component_names.get(comp_key, comp_key), str(max_w)]
            for idx_name in idx_names_in_scores:
                idx_score = scores.get(idx_name, scores.get(idx_name.lower(), {}))
                if not isinstance(idx_score, dict):
                    idx_score = {}
                comp_scores = idx_score.get("components", idx_score.get("breakdown", {}))
                if isinstance(comp_scores, dict):
                    val = comp_scores.get(comp_key, "—")
                    row.append(_fmt_num(val, decimals=1))
                else:
                    row.append("—")
            breakdown_data.append(row)

        # Totals row
        totals_row = ["TOTAL", "100"]
        for idx_name in idx_names_in_scores:
            idx_score = scores.get(idx_name, scores.get(idx_name.lower(), {}))
            if not isinstance(idx_score, dict):
                idx_score = {}
            t = _safe_get(idx_score, "total_score", _safe_get(idx_score, "total", "—"))
            totals_row.append(_fmt_num(t, decimals=1))
        breakdown_data.append(totals_row)

        col_w = [3.2 * cm, 2 * cm] + [3 * cm] * len(idx_names_in_scores)
        breakdown_table = _make_styled_table(breakdown_data, col_widths=col_w)

        # Highlight totals row
        total_row_idx = len(breakdown_data) - 1
        breakdown_table.setStyle(TableStyle([
            ("BACKGROUND", (0, total_row_idx), (-1, total_row_idx), Theme.BG_HEADER),
            ("TEXTCOLOR", (0, total_row_idx), (-1, total_row_idx), Theme.TEXT_LIGHT),
            ("FONTNAME", (0, total_row_idx), (-1, total_row_idx), "Helvetica-Bold"),
        ]))

        elements.append(breakdown_table)
        elements.append(Spacer(1, 0.6 * cm))

        # Cross-index consensus
        elements.append(Paragraph("Cross-Index Consensus", s["section_title"]))
        consensus = scores.get("consensus", scores.get("cross_index", {}))
        if isinstance(consensus, dict) and consensus:
            consensus_data = [["Metric", "Value"]]
            for k, v in consensus.items():
                consensus_data.append([str(k).replace("_", " ").title(), str(v)])
            elements.append(_make_styled_table(consensus_data, col_widths=[6 * cm, 10 * cm]))
        else:
            elements.append(Paragraph("No cross-index consensus data available.", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 3 — OI CHAIN ANALYSIS
    # ══════════════════════════════════════════════════════════════

    def _build_oi_chain_analysis(self, data: dict) -> list:
        """Build OI chain analysis page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("OI CHAIN ANALYSIS", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        oi_data = data.get("oi_data", {})
        if isinstance(oi_data, list):
            # Convert list of snapshots to dict keyed by index
            oi_dict = {}
            for snap in oi_data:
                if isinstance(snap, dict):
                    idx = snap.get("index_name", "NIFTY")
                    oi_dict[idx] = snap
            oi_data = oi_dict
        if not isinstance(oi_data, dict):
            oi_data = {}

        for idx_name in ["NIFTY", "BANKNIFTY"]:
            idx_oi = oi_data.get(idx_name, {})
            if not isinstance(idx_oi, dict):
                idx_oi = {}

            elements.append(Paragraph(f"{idx_name} OI Summary", s["section_title"]))

            # Summary KPIs
            summary_data = [
                ["Spot", "Max Pain", "PCR", "CE Wall", "PE Wall", "Buildup", "GEX"],
                [
                    _fmt_num(_safe_get(idx_oi, "spot_price"), decimals=1),
                    _fmt_num(_safe_get(idx_oi, "max_pain"), decimals=0),
                    _fmt_num(_safe_get(idx_oi, "pcr_overall", _safe_get(idx_oi, "pcr")), decimals=2),
                    f"{_fmt_num(_safe_get(idx_oi, 'ce_wall_strike'), decimals=0)} ({_fmt_num(_safe_get(idx_oi, 'ce_wall_oi'), decimals=0)})",
                    f"{_fmt_num(_safe_get(idx_oi, 'pe_wall_strike'), decimals=0)} ({_fmt_num(_safe_get(idx_oi, 'pe_wall_oi'), decimals=0)})",
                    str(_safe_get(idx_oi, "buildup_pattern", "—")),
                    _fmt_num(_safe_get(idx_oi, "gex_value"), decimals=1),
                ],
            ]
            elements.append(_make_styled_table(summary_data, col_widths=[2.2 * cm] * 7))
            elements.append(Spacer(1, 0.3 * cm))

            # OI chain table (top strikes)
            chain = idx_oi.get("chain_json") or idx_oi.get("chain") or []
            if isinstance(chain, dict):
                chain = chain.get("strikes", chain.get("data", []))

            if isinstance(chain, list) and chain:
                chain_header = ["Strike", "CE OI", "PE OI", "CE Chg", "PE Chg", "CE IV", "PE IV", "PCR"]
                chain_rows = [chain_header]
                display_chain = chain[:25] if len(chain) > 25 else chain
                for row in display_chain:
                    if isinstance(row, dict):
                        chain_rows.append([
                            _fmt_num(row.get("strike", row.get("strike_price")), decimals=0),
                            _fmt_num(row.get("ce_oi", 0), decimals=0),
                            _fmt_num(row.get("pe_oi", 0), decimals=0),
                            _fmt_num(row.get("ce_change", row.get("ce_oi_change", 0)), decimals=0),
                            _fmt_num(row.get("pe_change", row.get("pe_oi_change", 0)), decimals=0),
                            _fmt_num(row.get("ce_iv", 0), decimals=1),
                            _fmt_num(row.get("pe_iv", 0), decimals=1),
                            _fmt_num(
                                (float(row.get("pe_oi", 0) or 0) / float(row.get("ce_oi", 1) or 1))
                                if float(row.get("ce_oi", 0) or 0) > 0 else 0,
                                decimals=2,
                            ),
                        ])
                cw = [2 * cm] * 8
                elements.append(_make_styled_table(chain_rows, col_widths=cw))
            elements.append(Spacer(1, 0.4 * cm))

            # PCR chart
            chart = _create_pcr_chart(idx_oi)
            if chart:
                elements.append(chart)
                elements.append(Spacer(1, 0.3 * cm))

        # GEX interpretation
        gex_interp = _safe_get(oi_data.get("NIFTY", {}), "gex_interpretation")
        if gex_interp and gex_interp != "—":
            elements.append(Paragraph(f"GEX Interpretation: {gex_interp}", s["body_bold"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 4 — ASTRO DATA
    # ══════════════════════════════════════════════════════════════

    def _build_astro_data(self, data: dict) -> list:
        """Build astro data page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("ASTRO DATA", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        astro = data.get("astro_data", {})
        if isinstance(astro, list) and len(astro) > 0:
            astro = astro[0] if isinstance(astro[0], dict) else {}
        if not isinstance(astro, dict):
            astro = {}

        # Planet positions table
        elements.append(Paragraph("Planetary Positions (Sidereal / Lahiri)", s["section_title"]))

        planets = [
            ("Sun", "sun_deg", None),
            ("Moon", "moon_deg", None),
            ("Mercury", "mercury_deg", "mercury_retro"),
            ("Venus", "venus_deg", "venus_retro"),
            ("Mars", "mars_deg", "mars_retro"),
            ("Jupiter", "jupiter_deg", "jupiter_retro"),
            ("Saturn", "saturn_deg", "saturn_retro"),
            ("Rahu", "rahu_deg", None),
            ("Ketu", "ketu_deg", None),
        ]

        planet_data = [["Planet", "Degree", "Sign", "Retrograde"]]
        for planet_name, deg_key, retro_key in planets:
            deg = _safe_get(astro, deg_key)
            retro = "YES" if retro_key and _safe_get(astro, retro_key, False) else "No"
            if retro_key is None:
                retro = "N/A"
            sign = _degree_to_sign(deg) if deg != "—" else "—"
            planet_data.append([planet_name, _fmt_num(deg, decimals=2, suffix="\u00b0"), sign, retro])

        planet_table = _make_styled_table(
            planet_data, col_widths=[3 * cm, 3 * cm, 3.5 * cm, 3 * cm]
        )
        # Highlight retrograde rows
        for i, (_, _, retro_key) in enumerate(planets, 1):
            if retro_key and _safe_get(astro, retro_key, False):
                planet_table.setStyle(TableStyle([
                    ("TEXTCOLOR", (3, i), (3, i), Theme.RED),
                    ("FONTNAME", (3, i), (3, i), "Helvetica-Bold"),
                ]))
        elements.append(planet_table)
        elements.append(Spacer(1, 0.5 * cm))

        # Nakshatra, Tithi, Paksha
        elements.append(Paragraph("Nakshatra & Tithi", s["section_title"]))
        nak_data = [
            ["Metric", "Value"],
            ["Nakshatra", str(_safe_get(astro, "nakshatra"))],
            ["Pada", str(_safe_get(astro, "nakshatra_pada"))],
            ["Nature", str(_safe_get(astro, "nakshatra_nature"))],
            ["Tithi", f"{_safe_get(astro, 'tithi_name')} ({_safe_get(astro, 'tithi')})"],
            ["Tithi Nature", str(_safe_get(astro, "tithi_nature"))],
            ["Paksha", str(_safe_get(astro, "paksha"))],
            ["Yoga", str(_safe_get(astro, "current_yoga"))],
        ]
        elements.append(_make_styled_table(nak_data, col_widths=[5 * cm, 10 * cm]))
        elements.append(Spacer(1, 0.5 * cm))

        # Hora sequence
        elements.append(Paragraph("Hora Sequence", s["section_title"]))
        hora_seq = astro.get("hora_sequence_json") or astro.get("hora_sequence") or []
        if isinstance(hora_seq, list) and hora_seq:
            hora_header = ["#", "Time", "Planet", "Nature"]
            hora_rows = [hora_header]
            for i, hora in enumerate(hora_seq, 1):
                if isinstance(hora, dict):
                    hora_rows.append([
                        str(i),
                        str(hora.get("time", hora.get("start", "—"))),
                        str(hora.get("planet", hora.get("ruler", "—"))),
                        str(hora.get("nature", hora.get("bias", "—"))),
                    ])
                elif isinstance(hora, str):
                    hora_rows.append([str(i), "—", hora, "—"])
            elements.append(_make_styled_table(
                hora_rows, col_widths=[1.5 * cm, 4 * cm, 4 * cm, 4 * cm]
            ))
        else:
            elements.append(Paragraph("Hora sequence not available.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Active aspects
        elements.append(Paragraph("Active Planetary Aspects", s["section_title"]))
        aspects = astro.get("active_aspects_json") or astro.get("active_aspects") or []
        if isinstance(aspects, list) and aspects:
            aspect_header = ["Planet 1", "Aspect", "Planet 2", "Nature", "Orb"]
            aspect_rows = [aspect_header]
            for asp in aspects:
                if isinstance(asp, dict):
                    aspect_rows.append([
                        str(asp.get("planet1", asp.get("p1", "—"))),
                        str(asp.get("aspect", asp.get("type", "—"))),
                        str(asp.get("planet2", asp.get("p2", "—"))),
                        str(asp.get("nature", asp.get("bias", "—"))),
                        _fmt_num(asp.get("orb"), decimals=2, suffix="\u00b0"),
                    ])
            elements.append(_make_styled_table(
                aspect_rows, col_widths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 2.5 * cm]
            ))
        else:
            elements.append(Paragraph("No active aspects data available.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Astro score and bias
        elements.append(Paragraph("Astro Composite", s["section_title"]))
        astro_summary = [
            ["Metric", "Value"],
            ["Astro Score", _fmt_num(_safe_get(astro, "astro_score"), decimals=1)],
            ["Market Bias", str(_safe_get(astro, "market_bias"))],
        ]
        elements.append(_make_styled_table(astro_summary, col_widths=[5 * cm, 10 * cm]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 5 — GREEKS & IV
    # ══════════════════════════════════════════════════════════════

    def _build_greeks_iv(self, data: dict) -> list:
        """Build Greeks and IV page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("GREEKS & IMPLIED VOLATILITY", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        greeks = data.get("greeks_data", {})
        if isinstance(greeks, list) and len(greeks) > 0:
            greeks = greeks[0] if isinstance(greeks[0], dict) else {}
        if not isinstance(greeks, dict):
            greeks = {}

        # IV Rank
        elements.append(Paragraph("IV Rank", s["section_title"]))
        iv_rank_data = [
            ["Index", "IV Rank", "IV Percentile", "Current IV", "IV Mean"],
        ]
        for idx_name in ["NIFTY", "BANKNIFTY"]:
            idx_greeks = greeks.get(idx_name, greeks.get(idx_name.lower(), greeks))
            if not isinstance(idx_greeks, dict):
                idx_greeks = {}
            iv_rank_data.append([
                idx_name,
                _fmt_num(_safe_get(idx_greeks, "iv_rank"), decimals=1),
                _fmt_num(_safe_get(idx_greeks, "iv_percentile"), decimals=1),
                _fmt_num(_safe_get(idx_greeks, "current_iv", _safe_get(idx_greeks, "atm_iv")), decimals=2, suffix="%"),
                _fmt_num(_safe_get(idx_greeks, "iv_mean"), decimals=2, suffix="%"),
            ])
        elements.append(_make_styled_table(iv_rank_data, col_widths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]))
        elements.append(Spacer(1, 0.5 * cm))

        # ATM Greeks table
        elements.append(Paragraph("ATM Greeks", s["section_title"]))
        atm_header = ["Index", "Type", "Delta", "Gamma", "Theta", "Vega"]
        atm_data = [atm_header]
        for idx_name in ["NIFTY", "BANKNIFTY"]:
            idx_greeks = greeks.get(idx_name, greeks.get(idx_name.lower(), greeks))
            if not isinstance(idx_greeks, dict):
                idx_greeks = {}
            atm = idx_greeks.get("atm_greeks", idx_greeks.get("atm", {}))
            if not isinstance(atm, dict):
                atm = {}
            for opt_type in ["CE", "PE"]:
                type_data = atm.get(opt_type, atm.get(opt_type.lower(), {}))
                if not isinstance(type_data, dict):
                    type_data = {}
                atm_data.append([
                    idx_name,
                    opt_type,
                    _fmt_num(_safe_get(type_data, "delta"), decimals=4),
                    _fmt_num(_safe_get(type_data, "gamma"), decimals=6),
                    _fmt_num(_safe_get(type_data, "theta"), decimals=2),
                    _fmt_num(_safe_get(type_data, "vega"), decimals=2),
                ])
        elements.append(_make_styled_table(
            atm_data, col_widths=[2.5 * cm, 2 * cm, 2.5 * cm, 3 * cm, 2.5 * cm, 2.5 * cm]
        ))
        elements.append(Spacer(1, 0.5 * cm))

        # IV skew data
        elements.append(Paragraph("IV Skew Data", s["section_title"]))
        iv_skew = greeks.get("iv_skew") or greeks.get("iv_data") or []
        if isinstance(iv_skew, list) and iv_skew:
            skew_header = ["Strike", "CE IV", "PE IV", "Skew"]
            skew_rows = [skew_header]
            display_skew = iv_skew[:20] if len(iv_skew) > 20 else iv_skew
            for row in display_skew:
                if isinstance(row, dict):
                    ce_iv = float(row.get("ce_iv", row.get("call_iv", 0)) or 0)
                    pe_iv = float(row.get("pe_iv", row.get("put_iv", 0)) or 0)
                    skew = ce_iv - pe_iv if ce_iv and pe_iv else 0
                    skew_rows.append([
                        _fmt_num(row.get("strike", row.get("strike_price")), decimals=0),
                        _fmt_num(ce_iv, decimals=2, suffix="%"),
                        _fmt_num(pe_iv, decimals=2, suffix="%"),
                        _fmt_num(skew, decimals=2),
                    ])
            elements.append(_make_styled_table(
                skew_rows, col_widths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]
            ))
        else:
            elements.append(Paragraph("IV skew data not available.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # IV curve chart
        elements.append(Paragraph("IV Curve", s["section_title"]))
        iv_chart = _create_iv_curve_chart(greeks)
        if iv_chart:
            elements.append(iv_chart)
        else:
            elements.append(Paragraph("IV curve chart data not available.", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 6 — FII / DII DATA
    # ══════════════════════════════════════════════════════════════

    def _build_fii_dii(self, data: dict) -> list:
        """Build FII/DII data page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("FII / DII DATA", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        fii_dii = data.get("fii_dii_data", {})
        if isinstance(fii_dii, list) and len(fii_dii) > 0:
            fii_dii = fii_dii[0] if isinstance(fii_dii[0], dict) else {}
        if not isinstance(fii_dii, dict):
            fii_dii = {}

        # Cash segment
        elements.append(Paragraph("Cash Segment", s["section_title"]))
        cash_data = [
            ["Category", "Buy (Cr)", "Sell (Cr)", "Net (Cr)"],
            [
                "FII / FPI",
                _fmt_num(_safe_get(fii_dii, "fii_buy"), decimals=0),
                _fmt_num(_safe_get(fii_dii, "fii_sell"), decimals=0),
                _fmt_num(_safe_get(fii_dii, "fii_net"), decimals=0),
            ],
            [
                "DII",
                _fmt_num(_safe_get(fii_dii, "dii_buy"), decimals=0),
                _fmt_num(_safe_get(fii_dii, "dii_sell"), decimals=0),
                _fmt_num(_safe_get(fii_dii, "dii_net"), decimals=0),
            ],
        ]

        cash_table = _make_styled_table(cash_data, col_widths=[4 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm])
        # Color net columns
        for row_idx in [1, 2]:
            net_val = fii_dii.get("fii_net" if row_idx == 1 else "dii_net")
            try:
                color = Theme.GREEN if float(net_val) >= 0 else Theme.RED
                cash_table.setStyle(TableStyle([
                    ("TEXTCOLOR", (3, row_idx), (3, row_idx), color),
                    ("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]))
            except (ValueError, TypeError):
                pass
        elements.append(cash_table)
        elements.append(Spacer(1, 0.5 * cm))

        # FII Index Futures
        elements.append(Paragraph("FII Index Futures Position", s["section_title"]))
        fut_data = [
            ["Metric", "Value"],
            ["Long Contracts", _fmt_num(_safe_get(fii_dii, "fii_fut_long"), decimals=0)],
            ["Short Contracts", _fmt_num(_safe_get(fii_dii, "fii_fut_short"), decimals=0)],
            ["Net Contracts", _fmt_num(_safe_get(fii_dii, "fii_fut_net"), decimals=0)],
            ["Long/Short Ratio", _fmt_num(_safe_get(fii_dii, "fii_fut_ls_ratio"), decimals=2)],
            ["OI Change", _fmt_num(_safe_get(fii_dii, "fii_fut_oi_change"), decimals=0)],
        ]
        elements.append(_make_styled_table(fut_data, col_widths=[6 * cm, 8 * cm]))
        elements.append(Spacer(1, 0.5 * cm))

        # FII/DII chart
        elements.append(Paragraph("Net Flow Chart", s["section_title"]))
        chart = _create_fii_dii_chart(fii_dii)
        if chart:
            elements.append(chart)
        else:
            elements.append(Paragraph("Insufficient data for FII/DII chart.", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 7 — GLOBAL CUES
    # ══════════════════════════════════════════════════════════════

    def _build_global_cues(self, data: dict) -> list:
        """Build global cues page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("GLOBAL CUES", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        global_data = data.get("global_cues", {})
        if not isinstance(global_data, dict):
            global_data = {}

        # Gift Nifty
        elements.append(Paragraph("Gift Nifty", s["section_title"]))
        gift = global_data.get("gift_nifty", global_data.get("giftnifty", {}))
        if not isinstance(gift, dict):
            gift = {}
        gift_data = [
            ["Metric", "Value"],
            ["Last Price", _fmt_num(_safe_get(gift, "ltp", _safe_get(gift, "price")))],
            ["Change", _fmt_num(_safe_get(gift, "change"))],
            ["Change %", _fmt_num(_safe_get(gift, "change_pct"), suffix="%")],
            ["Premium to Nifty", _fmt_num(_safe_get(gift, "premium"))],
        ]
        elements.append(_make_styled_table(gift_data, col_widths=[6 * cm, 8 * cm]))
        elements.append(Spacer(1, 0.5 * cm))

        # US Markets
        elements.append(Paragraph("US Markets", s["section_title"]))
        us = global_data.get("us_markets", global_data.get("us", {}))
        if not isinstance(us, dict):
            us = {}
        us_indices = [
            ("Dow Jones", "dow", "djia"),
            ("S&P 500", "sp500", "spx"),
            ("Nasdaq", "nasdaq", "nq"),
        ]
        us_data = [["Index", "Close", "Change", "Change %"]]
        for display_name, key1, key2 in us_indices:
            idx = us.get(key1, us.get(key2, {}))
            if not isinstance(idx, dict):
                idx = {}
            us_data.append([
                display_name,
                _fmt_num(_safe_get(idx, "close", _safe_get(idx, "ltp"))),
                _fmt_num(_safe_get(idx, "change")),
                _fmt_num(_safe_get(idx, "change_pct"), suffix="%"),
            ])
        elements.append(_make_styled_table(
            us_data, col_widths=[4 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]
        ))
        elements.append(Spacer(1, 0.5 * cm))

        # Commodities
        elements.append(Paragraph("Commodities & Forex", s["section_title"]))
        commodities = global_data.get("commodities", {})
        if not isinstance(commodities, dict):
            commodities = {}
        forex = global_data.get("forex", {})
        if not isinstance(forex, dict):
            forex = {}

        commod_items = [
            ("Gold", commodities.get("gold", {})),
            ("Crude Oil", commodities.get("crude", commodities.get("crude_oil", {}))),
            ("Silver", commodities.get("silver", {})),
            ("USD/INR", forex.get("usdinr", forex.get("usd_inr", {}))),
        ]
        commod_data = [["Commodity", "Price", "Change", "Change %"]]
        for name, item in commod_items:
            if not isinstance(item, dict):
                item = {"price": item} if item else {}
            commod_data.append([
                name,
                _fmt_num(_safe_get(item, "price", _safe_get(item, "ltp", _safe_get(item, "close")))),
                _fmt_num(_safe_get(item, "change")),
                _fmt_num(_safe_get(item, "change_pct"), suffix="%"),
            ])
        elements.append(_make_styled_table(
            commod_data, col_widths=[4 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]
        ))
        elements.append(Spacer(1, 0.5 * cm))

        # Global sentiment
        elements.append(Paragraph("Global Sentiment Summary", s["section_title"]))
        sentiment = global_data.get("sentiment", global_data.get("summary", "—"))
        if isinstance(sentiment, dict):
            sent_data = [["Metric", "Value"]]
            for k, v in sentiment.items():
                sent_data.append([str(k).replace("_", " ").title(), str(v)])
            elements.append(_make_styled_table(sent_data, col_widths=[6 * cm, 10 * cm]))
        else:
            elements.append(Paragraph(f"Sentiment: {sentiment}", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 8 — SECTOR ANALYSIS
    # ══════════════════════════════════════════════════════════════

    def _build_sector_analysis(self, data: dict) -> list:
        """Build sector analysis page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("SECTOR ANALYSIS", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        sector_data = data.get("sector_data", {})
        if isinstance(sector_data, list):
            # Convert list to dict
            sd = {}
            for item in sector_data:
                if isinstance(item, dict):
                    name = item.get("sector_name", item.get("name", f"sector_{len(sd)}"))
                    sd[name] = item
            sector_data = sd
        if not isinstance(sector_data, dict):
            sector_data = {}

        # Sector table
        elements.append(Paragraph("Sector Heatmap", s["section_title"]))
        sector_header = ["Sector", "Change %", "Volume", "Impact on Nifty", "Weight"]
        sector_rows = [sector_header]

        from config import SECTORS as SECTOR_CONFIG

        leading = []
        lagging = []

        for sector_name in SECTOR_CONFIG.keys():
            sect = sector_data.get(sector_name, sector_data.get(sector_name.lower(), {}))
            if not isinstance(sect, dict):
                sect = {}
            chg = _safe_get(sect, "change_pct", 0)
            vol = _safe_get(sect, "volume")
            impact = _safe_get(sect, "impact_on_nifty")
            weight = SECTOR_CONFIG.get(sector_name, {}).get("weight_in_nifty", "—")

            try:
                chg_float = float(chg) if chg != "—" else 0
                if chg_float > 0.5:
                    leading.append(sector_name)
                elif chg_float < -0.5:
                    lagging.append(sector_name)
            except (ValueError, TypeError):
                chg_float = 0

            sector_rows.append([
                sector_name,
                _fmt_num(chg, decimals=2, suffix="%"),
                _fmt_num(vol, decimals=0),
                _fmt_num(impact, decimals=2),
                _fmt_num(weight, decimals=2) if weight != "—" else "—",
            ])

        sector_table = _make_styled_table(
            sector_rows, col_widths=[3 * cm, 2.5 * cm, 3 * cm, 3.5 * cm, 2.5 * cm]
        )
        # Color change% cells
        for i in range(1, len(sector_rows)):
            try:
                val = float(sector_rows[i][1].replace("%", "").replace(",", "").strip())
                color = Theme.GREEN if val >= 0 else Theme.RED
                sector_table.setStyle(TableStyle([
                    ("TEXTCOLOR", (1, i), (1, i), color),
                    ("FONTNAME", (1, i), (1, i), "Helvetica-Bold"),
                ]))
            except (ValueError, TypeError):
                pass
        elements.append(sector_table)
        elements.append(Spacer(1, 0.5 * cm))

        # Leading / Lagging
        elements.append(Paragraph("Leading & Lagging Sectors", s["section_title"]))
        ll_data = [
            ["Leading Sectors", "Lagging Sectors"],
            [
                ", ".join(leading) if leading else "None",
                ", ".join(lagging) if lagging else "None",
            ],
        ]
        elements.append(_make_styled_table(ll_data, col_widths=[8 * cm, 8 * cm]))
        elements.append(Spacer(1, 0.5 * cm))

        # Sector heatmap chart
        elements.append(Paragraph("Sector Performance Chart", s["section_title"]))
        chart = _create_sector_heatmap(sector_data)
        if chart:
            elements.append(chart)
        else:
            elements.append(Paragraph("Sector chart data not available.", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 9 — TRADE LOG
    # ══════════════════════════════════════════════════════════════

    def _build_trade_log(self, data: dict) -> list:
        """Build trade log page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("TRADE LOG", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        # Signals generated
        signals = data.get("signals", [])
        if not isinstance(signals, list):
            signals = [signals] if signals else []

        elements.append(Paragraph("Signals Generated Today", s["section_title"]))
        if signals:
            sig_header = ["Time", "Index", "Type", "Strike", "Score", "Confidence", "Status"]
            sig_rows = [sig_header]
            for sig in signals:
                if not isinstance(sig, dict):
                    continue
                sig_rows.append([
                    str(_safe_get(sig, "timestamp", _safe_get(sig, "time", "—")))[:19],
                    str(_safe_get(sig, "index_name", _safe_get(sig, "index"))),
                    str(_safe_get(sig, "signal_type", _safe_get(sig, "type"))),
                    _fmt_num(_safe_get(sig, "strike"), decimals=0),
                    _fmt_num(_safe_get(sig, "total_score", _safe_get(sig, "score")), decimals=1),
                    _fmt_num(_safe_get(sig, "confidence"), decimals=1),
                    str(_safe_get(sig, "status")),
                ])
            elements.append(_make_styled_table(
                sig_rows, col_widths=[2.8 * cm, 2.2 * cm, 1.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm]
            ))
        else:
            elements.append(Paragraph("No signals generated today.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Trades executed
        trades = data.get("trades", [])
        if not isinstance(trades, list):
            trades = [trades] if trades else []

        elements.append(Paragraph("Trades Executed", s["section_title"]))
        if trades:
            trade_header = ["Time", "Index", "Type", "Entry", "SL", "Target", "Exit", "P&L", "Status"]
            trade_rows = [trade_header]
            total_pnl = 0
            wins = 0
            losses = 0
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                pnl = trade.get("pnl", 0)
                try:
                    pnl_val = float(pnl or 0)
                    total_pnl += pnl_val
                    if pnl_val > 0:
                        wins += 1
                    elif pnl_val < 0:
                        losses += 1
                except (ValueError, TypeError):
                    pnl_val = 0

                trade_rows.append([
                    str(_safe_get(trade, "timestamp", _safe_get(trade, "time", "—")))[:19],
                    str(_safe_get(trade, "index_name", _safe_get(trade, "index"))),
                    str(_safe_get(trade, "signal_type", _safe_get(trade, "type"))),
                    _fmt_num(_safe_get(trade, "entry_price"), decimals=1),
                    _fmt_num(_safe_get(trade, "sl_price", _safe_get(trade, "sl")), decimals=1),
                    _fmt_num(_safe_get(trade, "target_price", _safe_get(trade, "target")), decimals=1),
                    _fmt_num(_safe_get(trade, "exit_price"), decimals=1),
                    _pnl_str(pnl),
                    str(_safe_get(trade, "status")),
                ])

            cw = [2.3 * cm, 1.8 * cm, 1.3 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2 * cm, 1.8 * cm]
            trade_table = _make_styled_table(trade_rows, col_widths=cw)
            # Color P&L column
            for i in range(1, len(trade_rows)):
                try:
                    pnl_text = trade_rows[i][7]
                    if pnl_text.startswith("+"):
                        color = Theme.GREEN
                    elif pnl_text.startswith("-"):
                        color = Theme.RED
                    else:
                        color = Theme.TEXT_DARK
                    trade_table.setStyle(TableStyle([
                        ("TEXTCOLOR", (7, i), (7, i), color),
                        ("FONTNAME", (7, i), (7, i), "Helvetica-Bold"),
                    ]))
                except (ValueError, TypeError, IndexError):
                    pass
            elements.append(trade_table)
            elements.append(Spacer(1, 0.4 * cm))

            # Win/loss summary
            elements.append(Paragraph("Trade Summary", s["section_title"]))
            summary_data = [
                ["Metric", "Value"],
                ["Total Trades", str(len(trades))],
                ["Winning Trades", str(wins)],
                ["Losing Trades", str(losses)],
                ["Win Rate", f"{(wins / len(trades) * 100):.1f}%" if trades else "—"],
                ["Total P&L", _pnl_str(total_pnl)],
            ]
            elements.append(_make_styled_table(summary_data, col_widths=[6 * cm, 8 * cm]))
        else:
            elements.append(Paragraph("No trades executed today.", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 10 — PRICE ACTION
    # ══════════════════════════════════════════════════════════════

    def _build_price_action(self, data: dict) -> list:
        """Build price action page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("PRICE ACTION", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        pa = data.get("price_action", {})
        if isinstance(pa, list) and len(pa) > 0:
            pa = pa[0] if isinstance(pa[0], dict) else {}
        if not isinstance(pa, dict):
            pa = {}

        for idx_name in ["NIFTY", "BANKNIFTY"]:
            idx_pa = pa.get(idx_name, pa.get(idx_name.lower(), pa))
            if not isinstance(idx_pa, dict):
                idx_pa = {}

            elements.append(Paragraph(f"{idx_name} Price Action", s["section_title"]))

            pa_data = [
                ["Metric", "Value"],
                ["EMA 9", _fmt_num(_safe_get(idx_pa, "ema_9", _safe_get(idx_pa, "ema9")), decimals=2)],
                ["EMA 21", _fmt_num(_safe_get(idx_pa, "ema_21", _safe_get(idx_pa, "ema21")), decimals=2)],
                ["EMA 50", _fmt_num(_safe_get(idx_pa, "ema_50", _safe_get(idx_pa, "ema50")), decimals=2)],
                ["RSI (14)", _fmt_num(_safe_get(idx_pa, "rsi", _safe_get(idx_pa, "rsi_14")), decimals=2)],
                ["VWAP", _fmt_num(_safe_get(idx_pa, "vwap"), decimals=2)],
                ["ATR", _fmt_num(_safe_get(idx_pa, "atr"), decimals=2)],
                ["Trend", str(_safe_get(idx_pa, "trend", _safe_get(idx_pa, "trend_classification")))],
            ]
            elements.append(_make_styled_table(pa_data, col_widths=[5 * cm, 10 * cm]))
            elements.append(Spacer(1, 0.3 * cm))

            # Support / Resistance
            elements.append(Paragraph(f"{idx_name} Support & Resistance", s["section_title"]))
            sr_data = [
                ["Level", "Type", "Strength"],
            ]
            supports = idx_pa.get("supports", idx_pa.get("support_levels", []))
            resistances = idx_pa.get("resistances", idx_pa.get("resistance_levels", []))

            if isinstance(supports, list):
                for sup in supports[:5]:
                    if isinstance(sup, dict):
                        sr_data.append([
                            _fmt_num(_safe_get(sup, "level", _safe_get(sup, "price")), decimals=1),
                            "Support",
                            str(_safe_get(sup, "strength", "—")),
                        ])
                    else:
                        sr_data.append([_fmt_num(sup, decimals=1), "Support", "—"])

            if isinstance(resistances, list):
                for res in resistances[:5]:
                    if isinstance(res, dict):
                        sr_data.append([
                            _fmt_num(_safe_get(res, "level", _safe_get(res, "price")), decimals=1),
                            "Resistance",
                            str(_safe_get(res, "strength", "—")),
                        ])
                    else:
                        sr_data.append([_fmt_num(res, decimals=1), "Resistance", "—"])

            if len(sr_data) > 1:
                sr_table = _make_styled_table(sr_data, col_widths=[5 * cm, 4 * cm, 4 * cm])
                # Color support green, resistance red
                for i in range(1, len(sr_data)):
                    color = Theme.GREEN if sr_data[i][1] == "Support" else Theme.RED
                    sr_table.setStyle(TableStyle([
                        ("TEXTCOLOR", (1, i), (1, i), color),
                        ("FONTNAME", (1, i), (1, i), "Helvetica-Bold"),
                    ]))
                elements.append(sr_table)
            else:
                elements.append(Paragraph("Support/Resistance levels not available.", s["body"]))
            elements.append(Spacer(1, 0.5 * cm))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 11 — WEEKLY PROBABILITY
    # ══════════════════════════════════════════════════════════════

    def _build_weekly_probability(self, data: dict) -> list:
        """Build weekly probability page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("WEEKLY PROBABILITY FORECAST", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        forecast = data.get("weekly_forecast", {})
        if isinstance(forecast, dict) and "days" in forecast:
            days = forecast["days"]
        elif isinstance(forecast, list):
            days = forecast
        else:
            days = []

        # Day-wise probabilities
        elements.append(Paragraph("Day-wise Forecast", s["section_title"]))
        if days and isinstance(days, list):
            fc_header = ["Day", "Date", "CE Prob", "PE Prob", "Neutral", "Bias", "Confidence"]
            fc_rows = [fc_header]
            for day in days:
                if not isinstance(day, dict):
                    continue
                fc_rows.append([
                    str(_safe_get(day, "day_of_week", _safe_get(day, "day"))),
                    str(_safe_get(day, "date")),
                    _fmt_num(_safe_get(day, "call_probability", _safe_get(day, "ce_prob")), decimals=1, suffix="%"),
                    _fmt_num(_safe_get(day, "put_probability", _safe_get(day, "pe_prob")), decimals=1, suffix="%"),
                    _fmt_num(_safe_get(day, "neutral_probability", _safe_get(day, "neutral_prob")), decimals=1, suffix="%"),
                    str(_safe_get(day, "predicted_bias", _safe_get(day, "bias"))),
                    _fmt_num(_safe_get(day, "confidence"), decimals=1, suffix="%"),
                ])
            fc_table = _make_styled_table(
                fc_rows, col_widths=[2.2 * cm, 2.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2.5 * cm, 2.2 * cm]
            )
            # Color bias cells
            for i in range(1, len(fc_rows)):
                try:
                    bias = fc_rows[i][5].upper()
                    if "BULL" in bias:
                        color = Theme.GREEN
                    elif "BEAR" in bias:
                        color = Theme.RED
                    else:
                        color = Theme.AMBER
                    fc_table.setStyle(TableStyle([
                        ("TEXTCOLOR", (5, i), (5, i), color),
                        ("FONTNAME", (5, i), (5, i), "Helvetica-Bold"),
                    ]))
                except (ValueError, TypeError, IndexError):
                    pass
            elements.append(fc_table)
        else:
            elements.append(Paragraph("Weekly forecast data not available.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Astro events for the week
        elements.append(Paragraph("Astro Events This Week", s["section_title"]))
        astro_events = []
        if isinstance(days, list):
            for day in days:
                if isinstance(day, dict):
                    events = day.get("astro_events_json", day.get("astro_events", []))
                    if isinstance(events, list):
                        for evt in events:
                            if isinstance(evt, dict):
                                astro_events.append(evt)
                            elif isinstance(evt, str):
                                astro_events.append({"event": evt, "date": _safe_get(day, "date")})

        if astro_events:
            evt_header = ["Date", "Event", "Impact"]
            evt_rows = [evt_header]
            for evt in astro_events:
                if isinstance(evt, dict):
                    evt_rows.append([
                        str(_safe_get(evt, "date")),
                        str(_safe_get(evt, "event", _safe_get(evt, "name"))),
                        str(_safe_get(evt, "impact", _safe_get(evt, "nature"))),
                    ])
            elements.append(_make_styled_table(
                evt_rows, col_widths=[3.5 * cm, 7 * cm, 4 * cm]
            ))
        else:
            elements.append(Paragraph("No specific astro events recorded this week.", s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # ML regime prediction
        elements.append(Paragraph("ML Regime Prediction", s["section_title"]))
        regime = forecast.get("ml_regime", forecast.get("regime", {})) if isinstance(forecast, dict) else {}
        if isinstance(regime, dict) and regime:
            regime_data = [["Metric", "Value"]]
            for k, v in regime.items():
                regime_data.append([str(k).replace("_", " ").title(), str(v)])
            elements.append(_make_styled_table(regime_data, col_widths=[6 * cm, 10 * cm]))
        else:
            elements.append(Paragraph("ML regime prediction not available.", s["body"]))

        return elements

    # ══════════════════════════════════════════════════════════════
    # PAGE 12 — RISK SUMMARY
    # ══════════════════════════════════════════════════════════════

    def _build_risk_summary(self, data: dict) -> list:
        """Build risk summary page."""
        elements = []
        s = self.styles

        elements.append(Paragraph("RISK SUMMARY", s["page_title"]))
        elements.append(Spacer(1, 0.3 * cm))

        risk = data.get("risk_summary", {})
        if isinstance(risk, list) and len(risk) > 0:
            risk = risk[0] if isinstance(risk[0], dict) else {}
        if not isinstance(risk, dict):
            risk = {}

        pnl_data = data.get("pnl_summary", {})
        if isinstance(pnl_data, list) and len(pnl_data) > 0:
            pnl_data = pnl_data[0] if isinstance(pnl_data[0], dict) else {}
        if not isinstance(pnl_data, dict):
            pnl_data = {}

        # Merge risk and pnl
        combined = {**pnl_data, **risk}

        # Daily P&L
        elements.append(Paragraph("Daily P&L", s["section_title"]))
        daily_pnl = combined.get("daily_pnl", combined.get("total_pnl", combined.get("net_pnl", "—")))
        pnl_table_data = [
            ["Metric", "Value"],
            ["Daily P&L", _pnl_str(daily_pnl)],
            ["Gross Profit", _fmt_num(combined.get("gross_profit", "—"))],
            ["Gross Loss", _fmt_num(combined.get("gross_loss", "—"))],
            ["Max Profit Trade", _fmt_num(combined.get("max_profit_trade", combined.get("best_trade", "—")))],
            ["Max Loss Trade", _fmt_num(combined.get("max_loss_trade", combined.get("worst_trade", "—")))],
        ]
        pnl_table = _make_styled_table(pnl_table_data, col_widths=[6 * cm, 8 * cm])
        # Color daily P&L
        try:
            color = _pnl_color(daily_pnl)
            pnl_table.setStyle(TableStyle([
                ("TEXTCOLOR", (1, 1), (1, 1), color),
                ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (1, 1), (1, 1), 11),
            ]))
        except Exception:
            pass
        elements.append(pnl_table)
        elements.append(Spacer(1, 0.5 * cm))

        # Risk checks
        elements.append(Paragraph("Risk Checks", s["section_title"]))

        from config import RISK_RULES
        risk_checks_data = [
            ["Rule", "Limit", "Actual", "Status"],
            [
                "Daily Loss Limit",
                f"{RISK_RULES['daily_loss_limit'] * 100:.1f}%",
                _fmt_num(combined.get("daily_loss_pct", "—"), decimals=2, suffix="%"),
                str(combined.get("daily_loss_status", "—")),
            ],
            [
                "Max Trades/Day",
                str(RISK_RULES["max_trades_per_day"]),
                str(combined.get("total_trades", combined.get("trades_taken", "—"))),
                str(combined.get("trade_limit_status", "—")),
            ],
            [
                "No Entry After",
                RISK_RULES["no_entry_after"],
                str(combined.get("last_entry_time", "—")),
                str(combined.get("time_limit_status", "—")),
            ],
            [
                "Consecutive Losses",
                str(RISK_RULES["consecutive_loss_pause"]),
                str(combined.get("consecutive_losses", "—")),
                str(combined.get("consecutive_loss_status", "—")),
            ],
            [
                "VIX Check",
                f"Reduce>{RISK_RULES['vix_reduce_above']}, Block>{RISK_RULES['vix_block_above']}",
                _fmt_num(combined.get("vix", combined.get("india_vix", "—")), decimals=2),
                str(combined.get("vix_status", "—")),
            ],
        ]
        risk_table = _make_styled_table(
            risk_checks_data, col_widths=[3.5 * cm, 3.5 * cm, 3.5 * cm, 3.5 * cm]
        )
        # Color status cells
        for i in range(1, len(risk_checks_data)):
            status = str(risk_checks_data[i][3]).upper()
            if "OK" in status or "PASS" in status or "GREEN" in status:
                color = Theme.GREEN
            elif "BREACH" in status or "FAIL" in status or "RED" in status or "BLOCK" in status:
                color = Theme.RED
            elif "WARN" in status or "YELLOW" in status:
                color = Theme.AMBER
            else:
                color = Theme.TEXT_DARK
            risk_table.setStyle(TableStyle([
                ("TEXTCOLOR", (3, i), (3, i), color),
                ("FONTNAME", (3, i), (3, i), "Helvetica-Bold"),
            ]))
        elements.append(risk_table)
        elements.append(Spacer(1, 0.5 * cm))

        # Drawdown
        elements.append(Paragraph("Drawdown & Capital", s["section_title"]))
        dd_data = [
            ["Metric", "Value"],
            ["Max Drawdown Today", _fmt_num(combined.get("max_drawdown", combined.get("max_dd", "—")))],
            ["Max Drawdown %", _fmt_num(combined.get("max_drawdown_pct", combined.get("max_dd_pct", "—")), suffix="%")],
            ["Peak Equity Today", _fmt_num(combined.get("peak_equity", "—"))],
            ["Lowest Equity Today", _fmt_num(combined.get("lowest_equity", combined.get("trough_equity", "—")))],
        ]
        elements.append(_make_styled_table(dd_data, col_widths=[6 * cm, 8 * cm]))
        elements.append(Spacer(1, 0.5 * cm))

        # Position sizing
        elements.append(Paragraph("Position Sizing & Capital Utilization", s["section_title"]))
        pos_data = [
            ["Metric", "Value"],
            ["Capital Deployed", _fmt_num(combined.get("capital_deployed", combined.get("capital_used", "—")))],
            ["Capital Available", _fmt_num(combined.get("capital_available", combined.get("free_capital", "—")))],
            ["Capital Utilization %", _fmt_num(combined.get("capital_utilization_pct", "—"), suffix="%")],
            ["Max Position Size", _fmt_num(combined.get("max_position_size", "—"))],
            ["Avg Position Size", _fmt_num(combined.get("avg_position_size", "—"))],
            ["Lots Used", str(combined.get("lots_used", combined.get("total_lots", "—")))],
        ]
        elements.append(_make_styled_table(pos_data, col_widths=[6 * cm, 8 * cm]))

        # Final note
        elements.append(Spacer(1, 1 * cm))
        elements.append(Paragraph(
            "This report is auto-generated by the AstroNifty Engine. "
            "All data is captured during live market hours and exported at market close.",
            s["small_note"],
        ))

        return elements
