"""Rendering is where untrusted document text meets the browser."""

from assistant.rendering import extract_cited_indices, render_answer


def test_script_tags_cannot_survive():
    html = render_answer("<script>alert('x')</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_attributes_cannot_be_injected_through_bold():
    html = render_answer('**a" onmouseover="evil()**')
    assert 'onmouseover="evil()"' not in html


def test_bold_and_citations_render():
    html = render_answer("Visiting is **14:00** [1].")
    assert "<strong>14:00</strong>" in html
    assert 'data-source="1"' in html


def test_a_list_after_a_lead_in_line_stays_a_list():
    html = render_answer("Bring these:\n- your letter\n- your medicines")
    assert "<p>Bring these:</p>" in html
    assert html.count("<li>") == 2


def test_blank_lines_separate_paragraphs():
    assert render_answer("One.\n\nTwo.").count("<p>") == 2


def test_wrapped_lines_join_into_one_paragraph():
    assert render_answer("One\ntwo").count("<p>") == 1


def test_cited_indices_are_unique_and_ordered():
    assert extract_cited_indices("a [2] b [1] c [2]") == [2, 1]


def test_empty_text_renders_nothing():
    assert render_answer("") == ""
