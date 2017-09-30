var codeMirrorContainer, headerCodeMirror = null, fragCodeMirror = null;
var fragHeaderLines;
var currentErrorPanel = null, errorWidgets = [];

function addSpan(elem, text, className) {
    var span = elem.appendChild(document.createElement("span"));
    span.innerHTML = text;
    span.className = className;
}

// fragHeaderLines is the number of
// lines in the fragment shader header.
function renderError(error, fragHeaderLines) {
    // Adds an error panel to the fragment code mirror.
    if (fragCodeMirror === null || currentErrorPanel !== null) return;

    var panel = document.createElement("div");
    panel.id = "panel-1";
    panel.className = "panel top";
    var close = panel.appendChild(document.createElement("a"));
    close.setAttribute("class", "remove-panel");
    close.innerHTML = "\u274c";
    CodeMirror.on(close, "click", function() {
        currentErrorPanel.clear();
        currentErrorPanel = null;
        // We won't get rid of the error widgets.
    });

    var label = panel.appendChild(document.createElement("span"));
    label.innerHTML = "The following errors were encountered during compilation:";
    panel.appendChild(document.createElement("br"));

    var errorLines = error.split("\n");
    var tmp, errorParts, lineNo, line, errorWidget;
    for (var i = 0; i < errorLines.length; ++i) {
        line = errorLines[i];
        if (line === "" || line === "\0") continue;

        // Add a span for each error.
        tmp = line.split(":");
        errorParts = tmp.splice(0,4);
        errorParts.push(tmp.join(" "));

        lineNo = parseInt(errorParts[2]) - fragHeaderLines;
        addSpan(panel, "Line " + lineNo, "error lineno");
        addSpan(panel, ": ", "error sep");
        addSpan(panel, errorParts[3], "error info");
        addSpan(panel, ": ", "error sep");
        addSpan(panel, errorParts[4], "error details");

        errorWidget = document.createElement("div");
        errorWidget.className = "errorwidget";
        addSpan(errorWidget, errorParts[3], "info");
        addSpan(errorWidget, ": ", "sep");
        addSpan(errorWidget, errorParts[4], "details");

        errorWidgets.push(fragCodeMirror.addLineWidget(lineNo-1, errorWidget));

        if (i < errorLines.length - 1) {
            panel.appendChild(document.createElement("br"));
        }
    }

    currentErrorPanel = fragCodeMirror.addPanel(panel, {position: "top", stable: true});
}

function loadCodeMirror() {
    codeMirrorContainer = document.getElementById('codemirror-container');

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

        elt.classList.toggle('smoothTransition');

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
    p.style.marginBottom = 10;
    p.innerHTML = "Fragment shader code:";
    codeMirrorContainer.appendChild(p);

    var placeholderElem = document.createElement('p');
    placeholderElem.innerHTML = 'Add fragment shader code here.';
    placeholderElem.style.color = 'grey';
    placeholderElem.style.margin = '0px';
    placeholderElem.style.fontSize = 12;

    fragCodeMirror = CodeMirror(function(elt) {
        elt.style.height = 640;
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
}
