"""
Regenerate the PDF and Word files in sample_data/.

The generated files are committed, so you only need to run this if you want to
change their contents. It exists because the corpus should include at least one
of every format the loaders claim to support — a PDF loader nothing ever feeds
a PDF is a loader you have not tested.

    .venv/bin/python scripts/build_binary_samples.py
"""

from pathlib import Path

import docx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"

# (heading, [paragraphs]) — headings become <h2>-equivalents in both formats.
PHARMACY = [
    (
        "Pharmacy and prescriptions",
        [
            "The hospital pharmacy is on the ground floor of the Green Zone, "
            "immediately past outpatient reception.",
        ],
    ),
    (
        "Opening hours",
        [
            "The pharmacy is open Monday to Friday from 09:00 to 18:00, and on Saturday "
            "from 09:00 to 13:00. It is closed on Sundays and bank holidays.",
            "Outside these hours, ward patients are supplied by the on-call pharmacist "
            "through the nurse in charge. Ask the ward, not the pharmacy counter.",
        ],
    ),
    (
        "Take-home medicines",
        [
            "When you are discharged, the ward sends your prescription to the pharmacy "
            "electronically. Dispensing takes between one and three hours depending on how "
            "busy the department is, and this is the single most common reason people wait "
            "to go home.",
            "You may wait in the discharge lounge on the ground floor of the Blue Zone. "
            "The pharmacy will telephone the lounge when your medicines are ready.",
        ],
    ),
    (
        "Prescription charges",
        [
            "Medicines given to you while you are an inpatient are free.",
            "Take-home medicines and outpatient prescriptions are charged at the standard "
            "NHS prescription rate, per item.",
            "You do not pay if you are under sixteen, aged sixteen to eighteen and in "
            "full-time education, over sixty, hold a valid medical or maternity exemption "
            "certificate, or hold a prepayment certificate. Bring the certificate with you.",
        ],
    ),
    (
        "Repeat prescriptions",
        [
            "The hospital does not issue repeat prescriptions. After your first supply, "
            "your GP takes over. The discharge summary sent to your GP lists everything you "
            "were given and for how long.",
            "Contact your GP practice about three days before your supply runs out.",
        ],
    ),
    (
        "Bringing your own medicines",
        [
            "Bring every medicine you take in its original box, including inhalers, eye "
            "drops, creams, herbal remedies, and anything bought over the counter.",
            "Your own medicines are checked by a pharmacist and, if they are suitable, used "
            "during your stay and returned to you when you leave.",
        ],
    ),
    (
        "Questions about your medicines",
        [
            "The medicines information line is 020 7946 0170, open 09:00 to 17:00 on "
            "weekdays. A pharmacist can explain what a medicine is for, how to take it, and "
            "what the common side effects are.",
            "If you think you are having a serious reaction to a medicine, do not wait for "
            "the information line. Contact your GP urgently, or in an emergency call 999.",
        ],
    ),
]

EMERGENCY = [
    (
        "Coming to the Emergency department",
        [
            "The Emergency department has its own entrance on Mill Lane and is open at all "
            "times. It is for serious injury and illness that cannot wait.",
        ],
    ),
    (
        "When to call 999 instead",
        [
            "Call 999 and do not drive yourself for chest pain, difficulty breathing, "
            "sudden weakness or numbness on one side, slurred speech, heavy bleeding that "
            "does not stop, a seizure that will not end, or loss of consciousness.",
        ],
    ),
    (
        "Where to go for something less serious",
        [
            "The urgent treatment centre next to the Emergency department handles sprains, "
            "simple fractures, minor burns, cuts needing stitches, and minor infections. It "
            "is open from 08:00 to 22:00 and the wait is usually much shorter.",
            "For advice at any hour, call NHS 111. For dental pain, contact a dentist; the "
            "Emergency department cannot treat toothache.",
        ],
    ),
    (
        "What happens when you arrive",
        [
            "You will be booked in at reception and then seen by a triage nurse, usually "
            "within fifteen minutes. Triage decides the order in which people are seen.",
            "Patients are seen in order of clinical need, not in order of arrival. This is "
            "why someone who arrived after you may be called first, and it is the part of "
            "the process people find hardest.",
        ],
    ),
    (
        "How long you may wait",
        [
            "Waiting times vary through the day and are usually longest on Monday mornings "
            "and between 18:00 and midnight. The screen in the waiting area shows the "
            "current average wait.",
            "Tell the reception desk if your condition gets worse while you are waiting.",
        ],
    ),
    (
        "What to bring",
        [
            "Bring a list of your medicines, the name of your GP practice, and your "
            "reading glasses. One person may stay with you.",
        ],
    ),
]


def build_pdf(path: Path, sections: list[tuple[str, list[str]]], title: str) -> None:
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(str(path), pagesize=A4, title=title)
    flow = []
    for index, (heading, paragraphs) in enumerate(sections):
        flow.append(Paragraph(heading, styles["Title" if index == 0 else "Heading2"]))
        flow.append(Spacer(1, 6))
        for paragraph in paragraphs:
            flow.append(Paragraph(paragraph, styles["BodyText"]))
            flow.append(Spacer(1, 6))
    document.build(flow)
    print(f"wrote {path}")


def build_docx(path: Path, sections: list[tuple[str, list[str]]]) -> None:
    document = docx.Document()
    for index, (heading, paragraphs) in enumerate(sections):
        document.add_heading(heading, level=1 if index == 0 else 2)
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
    document.save(str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    build_pdf(
        SAMPLE_DATA / "pharmacy-and-prescriptions.pdf", PHARMACY, "Pharmacy and prescriptions"
    )
    build_docx(SAMPLE_DATA / "emergency-department-what-to-expect.docx", EMERGENCY)
