from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

local_model_path = "Salesforce/codet5-base"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

code_snippet = """
	<java>
    @Override 
    String getAuthor(Document document) {
    	String allAuthor = "";
    	Elements contents = document.select(".typography_subtitle2__HAAtd.styles_postArticleAuthorInfo__XNgVX");
    	if (contents.isEmpty() != true) {
    		Elements author = contents.select("span");
        	if (author.text().contains("WRITTEN BY ")){
        		allAuthor = author.text().replace("WRITTEN BY ", "");
        	}
        	return allAuthor;
    	}
    	else {
    		contents = document.select(".elementor-element.elementor-element-b3d7692.elementor-widget.elementor-widget-heading");
    		allAuthor = contents.text();
    		if (allAuthor.contains("Written by ")){
        		allAuthor = allAuthor.replace("Written by ", "");
    		}
    		if (allAuthor.isEmpty() == true) {
    			allAuthor = "Anonymous";
    		}
    		return allAuthor;
    	}	
    }
"""

inputs = tokenizer.encode(code_snippet, return_tensors="pt")
print(tokenizer.decode(inputs[0]))

# Generate the output
outputs = model.generate(inputs, max_length=150, min_length=50, num_beams=50, early_stopping=True)
print(tokenizer.decode(outputs[0]))