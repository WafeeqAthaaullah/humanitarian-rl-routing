"""Generate a publication-ready technical report for the Dubai Humanitarian project.

Usage:
    python generate_project_pdf.py
    python generate_project_pdf.py --output results/humanitarian_rl_report.pdf

The report is intentionally generated from the repository configuration and
artifacts so that model, environment, and training settings remain traceable.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

import config


ROOT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#12304A")
BLUE = colors.HexColor("#1F6F8B")
TEAL = colors.HexColor("#2A9D8F")
GOLD = colors.HexColor("#E9C46A")
PALE_BLUE = colors.HexColor("#EAF3F7")
PALE_GOLD = colors.HexColor("#FFF7E1")
INK = colors.HexColor("#1D2935")
MUTED = colors.HexColor("#5C6B73")


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#D9EEF4"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=13.5,
            textColor=INK,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=12.5,
            leftIndent=12,
            firstLineIndent=-8,
            textColor=INK,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Metric",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            alignment=TA_CENTER,
            textColor=NAVY,
        )
    )
    return styles


def bullet(text: str, styles) -> Paragraph:
    return paragraph(f"• {text}", styles["BulletReport"])


def section_heading(number: str, title: str, styles) -> Paragraph:
    return paragraph(f"{number} &nbsp; {title}", styles["Section"])


def boxed_callout(title: str, body: str, styles, background=PALE_BLUE):
    table = Table(
        [[paragraph(f"<b>{title}</b><br/>{body}", styles["BodyReport"])]],
        colWidths=[174 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("LINEBEFORE", (0, 0), (0, 0), 4, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_table(rows: list[list[str]], widths: list[float], styles, header=True):
    converted = []
    for row_index, row in enumerate(rows):
        converted.append(
            [
                paragraph(
                    str(value),
                    styles["Subsection"] if header and row_index == 0 else styles["BodyReport"],
                )
                for value in row
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C6D5DC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def read_training_summary() -> tuple[int, float, float]:
    path = ROOT / "results" / "logs" / "training_metrics.csv"
    if not path.exists():
        return 0, 0.0, 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0, 0.0, 0.0
    return (
        len(rows),
        float(rows[-1]["epsilon"]),
        float(rows[-1]["lambda_penalty"]),
    )


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 11 * mm, width, 11 * mm, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(colors.HexColor("#D8E2E7"))
        canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, height - 11.5 * mm, "DUBAI HUMANITARIAN  |  DISTRIBUTIONAL RL")
    canvas.setStrokeColor(colors.HexColor("#D8E2E7"))
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "Humanitarian Supply Chain Transport-Mode Optimization")
    canvas.drawRightString(width - 18 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_story(styles):
    episodes, final_epsilon, final_lambda = read_training_summary()
    story = []

    cover = Table(
        [
            [
                paragraph(
                    "Humanitarian Supply Chain<br/>Transport-Mode Optimization<br/>via Distributional RL",
                    styles["ReportTitle"],
                )
            ],
            [paragraph("Technical project report | Dubai Humanitarian partnership", styles["Subtitle"])],
            [paragraph(f"Generated {datetime.now():%d %B %Y}", styles["Subtitle"])],
        ],
        colWidths=[174 * mm],
        rowHeights=[105 * mm, 12 * mm, 12 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("BOX", (0, 0), (-1, -1), 0, NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 20),
                ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            ]
        )
    )
    story.extend([Spacer(1, 18 * mm), cover, Spacer(1, 12 * mm)])
    story.append(
        boxed_callout(
            "Report scope",
            "This report summarizes the executable environment, agent, training, evaluation, and notebook artifacts in this repository. Values are traced to source files or the included/generated evaluation run; stochastic evaluation results can vary between runs.",
            styles,
            PALE_GOLD,
        )
    )
    story.append(PageBreak())

    story.append(section_heading("1", "Executive Summary & Problem Context", styles))
    story.append(
        paragraph(
            "In partnership with <b>Dubai Humanitarian</b>, this project formulates multi-echelon humanitarian dispatch as a constrained Markov Decision Process (CMDP). A shipment begins at an origin, may move through a regional checkpoint, and must reach destination inventory while balancing freight expenditure against emergency service reliability.",
            styles["BodyReport"],
        )
    )
    story.append(
        paragraph(
            "The operational challenge is deliberately adverse: capacity limits, a finite transport budget, stochastic route times, and disruption shocks interact with cargo urgency. Surface modes are economical but exposed to fat-tailed maritime delays; air is faster but expensive. The core safety objective is to keep Tier 1 emergency-cargo deadline violations below 1%, while minimizing transit cost.",
            styles["BodyReport"],
        )
    )
    metrics = Table(
        [
            [paragraph("SAFETY TARGET", styles["Small"]), paragraph("MODEL OUTPUT", styles["Small"]), paragraph("TRAINING SCALE", styles["Small"])],
            [paragraph("<b>&lt;1%</b><br/>Tier 1 deadline violations", styles["Metric"]), paragraph("<b>QR-DQN</b><br/>risk-sensitive policy", styles["Metric"]), paragraph(f"<b>{episodes or config.TRAIN_EPISODES:,}</b><br/>episodes in log/config", styles["Metric"])],
        ],
        colWidths=[58 * mm, 58 * mm, 58 * mm],
    )
    metrics.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE), ("GRID", (0, 0), (-1, -1), 0.35, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.extend([metrics, Spacer(1, 8)])

    story.append(section_heading("2", "MDP & Gymnasium Environment Architecture", styles))
    story.append(paragraph("The implementation in <b>envs/dubai_humanitarian_env.py</b> exposes a Gymnasium environment with an 11-dimensional Box observation and five discrete actions. The state is normalized before neural-network inference using the declared observation bounds.", styles["BodyReport"]))
    state_rows = [
        ["Index", "Feature", "Meaning in implementation"],
        ["0", "Cargo tier (u)", "1 = critical/Tier 1; 2 = standard/Tier 2."],
        ["1", "Volume (v)", "Randomized 1.0–100.0 units at reset."],
        ["2", "Weight (w)", "Randomized 0.5–50.0 tons; drives capacity and freight cost."],
        ["3", "Deadline (d_rem)", "Remaining days; initialized 5.0–30.0 and decreases with transit."],
        ["4", "Location (l)", "0 = origin, 1 = regional checkpoint, 2 = destination."],
        ["5", "Current mode (m_curr)", "0 = none, 2 = ocean, 3 = land."],
        ["6", "Air capacity", "Remaining normalized capacity; maximum physical capacity is 50 tons."],
        ["7", "Ocean capacity", "Remaining normalized capacity; maximum physical capacity is 500 tons."],
        ["8", "Land capacity", "Remaining normalized capacity; maximum physical capacity is 200 tons."],
        ["9", "Destination inventory (I_dest)", "Inventory buffer in days, initialized 0.0–100.0."],
        ["10", "Disruption probability (S_disrupt)", "Shock probability, initialized 0.0–0.5."],
    ]
    story.append(make_table(state_rows, [15 * mm, 39 * mm, 120 * mm], styles))
    story.append(Spacer(1, 6))
    action_rows = [
        ["Action", "Choice", "Legal transition"],
        ["0", "AIR_DIRECT", "Origin → destination; air capacity checked."],
        ["1", "OCEAN_TO_HUB", "Origin → regional checkpoint; ocean capacity checked."],
        ["2", "LAND_TO_HUB", "Origin → regional checkpoint; land capacity checked."],
        ["3", "SWITCH_TO_AIR", "Checkpoint → destination; air plus one switch day and cost."],
        ["4", "CONTINUE_NOMINAL", "Checkpoint → destination using the prior surface mode."],
    ]
    story.append(make_table(action_rows, [18 * mm, 42 * mm, 114 * mm], styles))
    story.append(Spacer(1, 6))
    story.append(boxed_callout("Legal transition masking", "At the origin, actions 3 and 4 are rejected. At the checkpoint, actions 0, 1, and 2 are rejected. Capacity checks additionally terminate infeasible actions with INVALID_ACTION_PENALTY = -100.0.", styles))
    story.append(paragraph("<b>Disruption and transit dynamics.</b> Each route samples a nominal Normal transit time with TRANSIT_STD = 0.5 days, bounded below by 1 day. When a shock occurs on an ocean route, the environment adds a LogNormal delay with mean parameter 2.0 and sigma 0.5, producing a fat-tailed maritime delay mixture. Base times are air 2, ocean leg 1 7, land leg 1 4, ocean leg 2 5, and land leg 2 3 days.", styles["BodyReport"]))

    story.append(PageBreak())
    story.append(section_heading("3", "Advanced RL Formulation & Methodology", styles))
    story.append(paragraph("<b>QR-DQN architecture.</b> QRDQNNetwork is a feed-forward network with two hidden ReLU layers of HIDDEN_DIM = 128. Its final layer emits ACTION_DIM × NUM_QUANTILES = 5 × 50 = 250 values, reshaped to [batch, action, quantile]. Quantile Huber loss, a target network, Adam optimization, observation normalization, and gradient clipping (norm 10.0) stabilize learning.", styles["BodyReport"]))
    story.append(paragraph(f"<b>Risk-sensitive action selection.</b> act_cvar sorts the predicted quantiles, takes the lowest cutoff fraction, averages those outcomes, and selects the action with the highest worst-case mean. The configured CVAR_ALPHA = {config.CVAR_ALPHA:.2f} uses the lowest {int(config.CVAR_ALPHA * config.NUM_QUANTILES)} of {config.NUM_QUANTILES} quantiles for conservative action selection.", styles["BodyReport"]))
    story.append(boxed_callout("Reproducibility note", f"Configured CVAR_ALPHA = {config.CVAR_ALPHA:.2f}; NUM_QUANTILES = {config.NUM_QUANTILES}. Training and evaluation checkpoints now record these values as metadata.", styles, PALE_GOLD))
    story.append(paragraph("<b>CMDP and dynamic Lagrangian penalty.</b> During training, Tier 1 is captured at reset. If a terminal transition misses its deadline, train.py subtracts the current lambda_penalty from the reward. After each episode, dual ascent updates λ as max(0, λ + LAMBDA_LR × (violation − SAFETY_BUDGET)); the repository sets LAMBDA_LR = 0.1 and SAFETY_BUDGET = 0.02. This makes repeated safety failures progressively more expensive to the learner.", styles["BodyReport"]))
    story.append(paragraph(f"The included training log contains {episodes:,} recorded episodes; its last recorded epsilon is {final_epsilon:.4f} and lambda_penalty is {final_lambda:.4f}. The training script is configured for {config.TRAIN_EPISODES:,} episodes.", styles["BodyReport"]))

    story.append(section_heading("4", "Benchmark Comparison & Empirical Results", styles))
    story.append(paragraph("The benchmark below is the output of scripts/evaluate_model.py executed against the checked-in checkpoint on 16 October 2026. The evaluator uses 500 Monte Carlo episodes, counts Tier 1 deadline misses, and estimates average cost by reversing reward normalization. Because the environment is stochastic and unseeded in the evaluator, reruns may differ.", styles["BodyReport"]))
    benchmark_rows = [
        ["Policy", "Tier 1 violation rate", "Average cost / episode", "Decision rule"],
        ["Greedy-Cheapest", "42.7%", "$88,963.30", "Ocean first, then nominal continuation; capacity fallback."],
        ["Static Threshold", "6.0%", "$29,064.54", "Air when Tier 1 and d_rem < 12 days; otherwise ocean/continue."],
        ["QR-DQN (ours)", "6.6%", "$15,302.43", "Learned distributional policy with CVaR action selection."],
    ]
    story.append(make_table(benchmark_rows, [35 * mm, 35 * mm, 38 * mm, 68 * mm], styles))
    story.append(Spacer(1, 6))
    story.append(boxed_callout("Interpretation", "In the captured run, QR-DQN achieved the lowest average freight cost and improved substantially over both heuristics on cost, but it did not reproduce the requested 0.0% violation / $500 claim: the observed result was 6.6% / $15,302.43. Those requested figures should be treated as a target or external result until a seeded, reproducible run produces them.", styles, PALE_GOLD))
    comparison_plot = ROOT / "results" / "plots" / "performance_comparison.png"
    if comparison_plot.exists():
        story.append(Image(str(comparison_plot), width=153 * mm, height=92 * mm))
        story.append(paragraph("Figure 1. Evaluation comparison generated by evaluate_model.py.", styles["Small"]))

    story.append(PageBreak())
    story.append(section_heading("5", "Live Demo & System Execution", styles))
    story.append(paragraph("The workspace is modular: <b>envs/</b> owns the Gymnasium simulation and transition dynamics; <b>agents/</b> contains QR-DQN, replay memory, CVaR selection, and heuristic baselines; <b>scripts/</b> contains training, evaluation, and the interactive command-line demo. config.py is the single source of environment constants and hyperparameters.", styles["BodyReport"]))
    structure_rows = [
        ["Component", "Execution responsibility"],
        ["scripts/train.py", "Runs episodes, applies the dynamic safety multiplier, trains from replay, updates the target network, and writes logs/plots."],
        ["results/checkpoints/qr_dqn_model.pth", "Serialized QR-DQN network weights loaded by evaluate_model.py."],
        ["scripts/evaluate_model.py", "Runs 500 episodes for Greedy-Cheapest, Static Threshold, and QR-DQN; prints metrics and saves the comparison plot."],
        ["project_demonstration.ipynb", "Interactive narrative covering the problem, CMDP state/action design, QR-DQN/CVaR method, and evaluation visualization."],
    ]
    story.append(make_table(structure_rows, [55 * mm, 121 * mm], styles))
    story.append(Spacer(1, 8))
    story.append(boxed_callout("Run sequence", "Install dependencies, train or reuse the checkpoint, run the benchmark, then open the notebook or demo for an interactive trace. The report generator itself only reads project artifacts and does not retrain the model.", styles))
    story.append(Spacer(1, 6))
    story.append(paragraph("<b>Recommended commands</b>", styles["Subsection"]))
    story.append(paragraph("pip install -r requirements.txt<br/>python scripts/train.py<br/>python scripts/evaluate_model.py<br/>python scripts/demo.py<br/>python generate_project_pdf.py", styles["BodyReport"]))

    story.append(section_heading("6", "Traceability & Limitations", styles))
    story.append(paragraph("Primary sources inspected: envs/dubai_humanitarian_env.py, agents/qr_dqn_agent.py, agents/heuristics.py, scripts/train.py, scripts/evaluate_model.py, project_demonstration.ipynb, config.py, README.md, and results/logs/training_metrics.csv. The model checkpoint and plots are treated as generated artifacts rather than independently reinterpreted.", styles["BodyReport"]))
    story.append(bullet("The evaluator does not seed env.reset(), so benchmark values are stochastic.", styles))
    story.append(bullet("The configured CVaR fraction is 5%, despite the 10% wording in comments and the requested brief.", styles))
    story.append(bullet("The included evaluation run did not meet the stated <1% safety target; further training, calibration, seeding, and/or constraint tuning would be needed before making a verified safety claim.", styles))
    story.append(Spacer(1, 10))
    story.append(boxed_callout("Publication readiness", "For a final paper, report a fixed seed, confidence intervals over repeated evaluation seeds, the exact checkpoint provenance, and a reconciled CVaR α setting. This document preserves the current repository evidence so those follow-up claims remain auditable.", styles))
    return story


def generate(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = BaseDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=20 * mm,
        title="Humanitarian Supply Chain Transport-Mode Optimization via Distributional RL",
        author="Dubai Humanitarian RL Routing Project",
    )
    doc.addPageTemplates([PageTemplate(id="report", frames=[], onPage=on_page)])
    doc.pageTemplates[0].frames = [
        Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id="normal",
            showBoundary=0,
        )
    ]
    doc.build(build_story(styles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "humanitarian_rl_transport_optimization_report.pdf",
        help="Output PDF path.",
    )
    args = parser.parse_args()
    generate(args.output if args.output.is_absolute() else ROOT / args.output)
    print(f"PDF report written to {args.output if args.output.is_absolute() else ROOT / args.output}")


if __name__ == "__main__":
    main()
