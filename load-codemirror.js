var codeMirrorContainer = document.getElementById('codemirror-container');

headerCodeMirror = CodeMirror(function(elt) {
	var toggleButton = document.createElement('button');
	toggleButton.className = 'center';
	toggleButton.innerHTML = 'Hide header file';
	toggleButton.onclick = function() {
		elt.classList.toggle('hide');
		if (elt.classList.contains('hide')) {
			this.innerHTML = 'Show header file';
		} else {
			this.innerHTML = 'Hide header file';
		}
	};

	codeMirrorContainer.appendChild(toggleButton);
	codeMirrorContainer.appendChild(elt);
}, {
	value: '// Loading header file...',
	lineNumbers: true,
	lineWrapping: true,
	gutter: true,
	readOnly: true,
	matchBrackets: true,
	scrollbarStyle: "overlay",
	mode: 'x-shader/x-fragment',
	indentUnit: 4,
	theme: 'monokai'
});

var p = document.createElement('p');
p.style.marginBottom = 0;
p.innerHTML = "Fragment shader code:";
codeMirrorContainer.appendChild(p);

var placeholderElem = document.createElement('p');
placeholderElem.innerHTML = 'Add fragment shader code here.';
placeholderElem.style.color = 'grey';
placeholderElem.style.margin = '0px';
placeholderElem.style.fontSize = 12;

fragCodeMirror = CodeMirror(function(elt) {
	codeMirrorContainer.appendChild(elt);
}, {
	value: '// Loading fragment shader source...',
	lineNumbers: true,
	lineWrapping: true,
	foldGutter: true,
	gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"],
	placeholder: placeholderElem,
	matchBrackets: true,
	scrollbarStyle: "overlay",
	mode: 'x-shader/x-fragment',
	indentUnit: 4,
	theme: 'monokai'
});

var fragHeaderReq = new XMLHttpRequest();
fragHeaderReq.onload = function() {
	var fragHeaderSrc = this.response;

	headerCodeMirror.setValue(fragHeaderSrc);
	headerCodeMirror.clearHistory();

	var fragReq = new XMLHttpRequest();
	fragReq.onload = function() {
		var fragSrc = this.response;

		fragCodeMirror.setValue(fragSrc);
		fragCodeMirror.clearHistory();

		init();
	};
	fragReq.open("GET", "fragment-source.glsl");
	fragReq.responseType = "text";
	fragReq.send();
};
fragHeaderReq.open("GET", "fragment-header.glsl");
fragHeaderReq.responseType = "text";
fragHeaderReq.send();
