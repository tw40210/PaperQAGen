from src.qa_gpt.core.utils.pdf_processor import extract_markdown_elements


def test_extract_markdown_elements_basic():
    """Test basic extraction of images and tables"""
    test_text = """
    This is a test document.
    ![](images/test1.jpg)
    Some content here.
    <html><body><table>
        <tr><td>Test</td></tr>
    </table></body></html>
    """

    images, tables, modified_text = extract_markdown_elements(test_text)

    # Check images
    assert len(images) == 1
    assert images[0] == "images/test1.jpg"

    # Check tables
    assert len(tables) == 1
    assert "<table>" in tables[0]

    # Check modified text
    assert "<<%%image_1>>" in modified_text
    assert "<<%%table_1>>" in modified_text
    assert "test1.jpg" not in modified_text
    assert "<table>" not in modified_text


def test_extract_markdown_elements_multiple():
    """Test extraction of multiple images and tables"""
    test_text = r"""
# 2 Single-Task Benchmark  

We start with model performances on the WebApplK benchmark. As illustrated in Tab. 1, each challenge of the benchmark focuses on a single task described by two test cases, one success and one failure. The task is about completing an atomic action (e.g. submitting a form, retrieving all posts),involving user interactions and access to a mocked API. More details of the benchmark can be found at [Cui, 2024b].  

![](images/55003e5596fc70c07d7c56961bab5d34ed7bbeb6c11be4d18aa74d27c87599bd.jpg)  
Table 1: Illustration of WebApplK Test Cases   
Table 2: WebApplK: pass@ 1 Results for Selected Models  

The prompt is straightforward: we feed test files to the model, expecting it to generate code passing these tests.  

Generate TaskA.js to pass the tests below: ${T a b.1(a)}${T a b.1(b)}$ . RETURN CODE ONLY.  

The resulting lines of code is typically between 40 and 50.  

# 2.1 Results  

Due to budget constraints, we only obtained $p a s s@1$ results for the ol models. Nevertheless, as shown in Tab. 2, they demonstrate impressive performance, lifting SOTA by $7\%$  

<html><body><table><tr><td>Model</td><td>pass@1</td></tr><tr><td>ol-preview</td><td>0.952</td></tr><tr><td>o1-mini</td><td>0.939</td></tr><tr><td>gpt-4o-2024-08-06</td><td>0.885</td></tr><tr><td>claude-3.5-sonnet</td><td>0.881</td></tr><tr><td>deepseek-v2.5</td><td>0.834</td></tr><tr><td> mistral-large-2</td><td>0.780</td></tr></table></body></html>  

As part of this achievement, the two ol models unlock a total of 16 challenges never solved by previous non-reasoning models. Next, we pick two examples to illustrate how reasoning models solve them.  
    """

    images, tables, modified_text = extract_markdown_elements(test_text)

    # Check images
    assert len(images) == 1
    assert (
        images[0] == "images/55003e5596fc70c07d7c56961bab5d34ed7bbeb6c11be4d18aa74d27c87599bd.jpg"
    )

    # Check tables
    assert len(tables) == 1
    table = tables[0].strip()  # Remove trailing whitespace

    # Check table content
    assert "<table>" in table
    assert "<tr><td>Model</td><td>pass@1</td></tr>" in table
    assert "<td>ol-preview</td><td>0.952</td>" in table
    assert (
        "<td> mistral-large-2</td><td>0.780</td>" in table
    )  # Note the space before mistral-large-2

    # Check modified text
    assert "<<%%image_1>>" in modified_text
    assert "<<%%table_1>>" in modified_text


def test_extract_markdown_elements_no_matches():
    """Test text with no images or tables"""
    test_text = "Just plain text with no images or tables"

    images, tables, modified_text = extract_markdown_elements(test_text)

    assert len(images) == 0
    assert len(tables) == 0
    assert modified_text == test_text


def test_extract_markdown_elements_non_matching_images():
    """Test images not in images directory"""
    test_text = "![](other/path/image.jpg)"

    images, tables, modified_text = extract_markdown_elements(test_text)

    assert len(images) == 0
    assert len(tables) == 0
    assert modified_text == test_text
