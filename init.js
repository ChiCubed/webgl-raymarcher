var glContext, glCanvas;
// Vertex Buffer Object
var vbo;
// WebGL shader program
var program = null;
// Uniforms - values which are constant during
// each frame's rendering.
var cameraPosUniform, viewToWorldUniform, screenSizeUniform,
    timeUniform, frequencyDataUniform;

var fpsElement, audioElement, audioFileInput;

// ID of next frame to render.
var nextRenderFrame = null;
// The time at which the current render loop started.
var startTime;
// The time at which the last frame was rendered.
var lastTime;

// Temporary variable for camera movement.
// Used by 'handleKeyDown'.
var camDisp;

// Stores which keys are pressed.
var keys = {};

// time counters
var currentTime, deltaTime;

// Variables for use by the handleKeyDown function.
var forwardsVec, backwardsVec, rightVec, leftVec;

// The vertex shader source.
var vertexSrc = "// Vertex shader source.                    \n"+
                "                                            \n"+
                "// current vertex position                  \n"+
                "attribute vec2 position;                    \n"+
                "                                            \n"+
                "void main() {                               \n"+
                "    // gl_Position must be set              \n"+
                "    // by the vertex shader                 \n"+
                "    gl_Position = vec4(position, 0.0, 1.0); \n"+
                "}                                           \n";

// Angle in terms of yaw, pitch, roll
var angle, cameraPos;
var viewToWorldMat;

// Texture for transmitting audio frequency
// data.
var frequencyTexture;

var AudioContext;
var audioCtx, source, analyser, stream;

var microphoneSource, audioElementSource;

var fftResult;


function createShader(gl, type, source) {
	var shader = gl.createShader(type);

	// Load the source into the shader
	gl.shaderSource(shader, source);

	gl.compileShader(shader);

	// We only return the shader if it
	// compiled successfully.
	var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
	if (success) return shader;

	// if the WebGL context wasn't lost:
	if (!gl.isContextLost()) {
		// Otherwise we log what went wrong.
        var error = gl.getShaderInfoLog(shader);
		console.log(error);

		gl.deleteShader(shader);

        return error;
	}
}


function createProgram(gl, vertexShader, fragmentShader) {
	var program = gl.createProgram();

	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);

	gl.linkProgram(program);

	var success = gl.getProgramParameter(program, gl.LINK_STATUS);
	if (success) return program;

	if (!gl.isContextLost()) {
        var error = gl.getProgramInfoLog(program);
		console.log(error);

		gl.deleteProgram(program);

        return null;
	}
}

function generateProgramFromSources(vertSrc, fragSrc) {
    // First, get rid of error widgets.
    if (currentErrorPanel !== null) {
        currentErrorPanel.clear();
        currentErrorPanel = null;
    }

    for (var i = 0; i < errorWidgets.length; ++i) {
        errorWidgets[i].clear();
    }
    errorWidgets = [];

    // Now, actually compile the shaders.
    var vertShader = createShader(glContext, glContext.VERTEX_SHADER, vertSrc);
	var fragShader = createShader(glContext, glContext.FRAGMENT_SHADER, fragSrc);

    if (fragShader instanceof String || typeof fragShader === "string") {
        renderError(fragShader, headerCodeMirror.lineCount());
        return null; // compilation failed
    }

	// Now we 'link' them into a program
	// which can be used by WebGL.
	return createProgram(glContext, vertShader, fragShader);
}


function handleMouseMove(evt) {
    angle[0] -= evt.movementX / 300;
    angle[1] -= evt.movementY / 300;
}

function handleKeyDown(evt) {
    keys[evt.keyCode] = true;
    evt.preventDefault();
}

function handleKeyUp(evt) {
    keys[evt.keyCode] = false;
    evt.preventDefault();
}

function changePointerLock() {
    if (document.pointerLockElement === glCanvas ||
        document.mozPointerLockElement === glCanvas) {
        document.addEventListener("mousemove", handleMouseMove, false);
        document.addEventListener("keydown", handleKeyDown, false);
        document.addEventListener("keyup", handleKeyUp, false);
    } else {
        document.removeEventListener("mousemove", handleMouseMove, false);
        document.removeEventListener("keydown", handleKeyDown, false);
        document.removeEventListener("keyup", handleKeyUp, false);
        keys = {};
    }
}

function resetCamera() {
    angle = [Math.PI, -0.2, 0.0];
    cameraPos = [0.0, 3.0, -10.0];
}

function writeToFFTTexture() {
    // Get data
    analyser.getByteFrequencyData(fftResult);

    // Write data
    glContext.bindTexture(glContext.TEXTURE_2D, frequencyTexture);
    glContext.texImage2D(glContext.TEXTURE_2D, 0, glContext.LUMINANCE,
                         fftResult.length, 1, 0, glContext.LUMINANCE, glContext.UNSIGNED_BYTE,
                         fftResult);
}


function resizeCanvas(width, height) {
    // Resizes the canvas.
    glCanvas.width = width;
    glCanvas.height = height;
    glCanvas.style.maxWidth = width;

    glContext.viewport(0, 0, width, height);
}

function changeAudioInput(inputMethod) {
    if (inputMethod == 'microphone') {
        navigator.mediaDevices.getUserMedia({audio: true, video: false})
            .then(function(stream) {
            source.disconnect();
            analyser.disconnect();

            source = microphoneSource;
            source.connect(analyser);
        });
        audioElement.src = '';
    } else if (inputMethod == 'file') {
        audioElement.src = URL.createObjectURL(audioFileInput.files[0]);

        source.disconnect();
        analyser.disconnect();

        source = audioElementSource;
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
    }
}

// Note: This function is called by load-codemirror.js
// when all the files have been loaded.
function init() {
	glCanvas = document.getElementById('gl-canvas');
    fpsElement = document.getElementById('fps-counter');
    audioElement = document.getElementById('audio-player');
    audioFileInput = document.getElementById('audio-file');

    audioFileInput.onchange = function() {
        changeAudioInput('file');

        // This ensures that the
        // onchange selector will
        // execute even if the same
        // file is selected twice,
        // e.g.:
        // select a file,
        // change audio input,
        // select the same file.
        this.value = null;
        return false;
    };

	// Get WebGL context.
	glContext = glCanvas.getContext('webgl') || glCanvas.getContext('experimental-webgl');

	if (!glContext) {
		alert('Failed to initialise WebGL.');
	}

    // Create the two triangles which will be used
    // to draw our scene on.

    var vertices = [
        -1.0, 1.0, // top left
         1.0, 1.0, // top right
         1.0,-1.0, // bottom right
        -1.0, 1.0, // top left
         1.0,-1.0, // bottom right
        -1.0,-1.0  // bottom left
    ];

    vbo = glContext.createBuffer();

    viewToWorldMat = mat4.create();
    camDisp = vec3.create();

    forwardsVec = vec3.fromValues(0, 0, -1);
    backwardsVec = vec3.fromValues(0, 0, 1);
    rightVec = vec3.fromValues(1, 0, 0);
    leftVec = vec3.fromValues(-1, 0, 0);

	// gl.ARRAY_BUFFER is a 'bind point'
	// for WebGL, which indicates where
	// the data is located.
	glContext.bindBuffer(glContext.ARRAY_BUFFER, vbo);

	// We convert the positions to a 32-bit float array.
	// gl.STATIC_DRAW indicates that the plane
	// will not move during the render loop.
	glContext.bufferData(glContext.ARRAY_BUFFER, new Float32Array(vertices), glContext.STATIC_DRAW);

    glCanvas.requestPointerLock = glCanvas.requestPointerLock ||
                               glCanvas.mozRequestPointerLock;

    glCanvas.addEventListener("click", function() {
        glCanvas.requestPointerLock();
    }, false);

    document.addEventListener("pointerlockchange", changePointerLock, false);
    document.addEventListener("mozpointerlockchange", changePointerLock, false);

    // Web Audio initialisation
    // https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Visualizations_with_Web_Audio_API

    AudioContext = window.AudioContext || window.webkitAudioContext;

    navigator.mediaDevices.getUserMedia({audio: true, video: false})
        .then(function(stream) {
        audioCtx = new AudioContext();
        analyser = audioCtx.createAnalyser();

        microphoneSource = audioCtx.createMediaStreamSource(stream);
        audioElementSource = audioCtx.createMediaElementSource(audioElement);

        source = microphoneSource;
        source.connect(analyser);

        analyser.fftSize = 1024;
        var bufferLength = analyser.frequencyBinCount;
        fftResult = new Uint8Array(bufferLength);

        // Frequency texture initialisation
        frequencyTexture = glContext.createTexture();

        glContext.bindTexture(glContext.TEXTURE_2D, frequencyTexture);

        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MIN_FILTER, glContext.NEAREST);
        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MAG_FILTER, glContext.NEAREST);
        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_S, glContext.CLAMP_TO_EDGE);
        glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_T, glContext.CLAMP_TO_EDGE);

        // Sets the unpack alignment for the data.
        glContext.pixelStorei(glContext.UNPACK_ALIGNMENT, 1);

        recompileShader();
    });
}

function setViewToWorld() {
	mat4.identity(viewToWorldMat);
    // We rotate about the Y axis, then the X axis, then the Z axis.
	mat4.rotateY(viewToWorldMat, viewToWorldMat, angle[0]);
	mat4.rotateX(viewToWorldMat, viewToWorldMat, angle[1]);
	mat4.rotateZ(viewToWorldMat, viewToWorldMat, angle[2]);
}

function getProgramAttribLocations() {
	// The shader program now needs to know
	// where the data being used in the
	// vertex shader (namely the
	// position attribute) is coming from.
	// We set that here.
	var positionAttribLoc = glContext.getAttribLocation(program, 'position');
	glContext.enableVertexAttribArray(positionAttribLoc);

	// We now tell WebGL how to extract
	// data out of the array verticesBuffer
	// and give it to the vertex shader.
	// The three main arguments are the
	// first three. In order these indicate:
	// 1.  where to bind the current ARRAY_BUFFER to
	// 2.  how many components there are per attribute
	//       (in this case two)
	// 3.  the type of the data
	glContext.vertexAttribPointer(positionAttribLoc, 2, glContext.FLOAT, false, 0, 0);


	// We also get the uniform locations,
	// to pass data to/from the shader.
	cameraPosUniform = glContext.getUniformLocation(program, "cameraPos");
	viewToWorldUniform = glContext.getUniformLocation(program, "viewToWorld");
	screenSizeUniform = glContext.getUniformLocation(program, "screenSize");
	timeUniform = glContext.getUniformLocation(program, "time");
    frequencyDataUniform = glContext.getUniformLocation(program, "frequencyData");
}

function moveCamera() {
    // Move camera based on current keypresses.

    var movingUp = keys[87] || keys[38];
    var movingDown = keys[83] || keys[40];
    var movingLeft = keys[65] || keys[37];
    var movingRight = keys[68] || keys[39];

    if (movingUp && !movingDown) {
        vec3.transformMat4(camDisp, forwardsVec, viewToWorldMat);
        vec3.scale(camDisp, camDisp, deltaTime / 120);
        vec3.add(cameraPos, cameraPos, camDisp);
    }
    if (movingDown && !movingUp) {
        vec3.transformMat4(camDisp, backwardsVec, viewToWorldMat);
        vec3.scale(camDisp, camDisp, deltaTime / 120);
        vec3.add(cameraPos, cameraPos, camDisp);
    }
    if (movingLeft && !movingRight) {
        vec3.transformMat4(camDisp, leftVec, viewToWorldMat);
        vec3.scale(camDisp, camDisp, deltaTime / 120);
        vec3.add(cameraPos, cameraPos, camDisp);
    }
    if (movingRight && !movingLeft) {
        vec3.transformMat4(camDisp, rightVec, viewToWorldMat);
        vec3.scale(camDisp, camDisp, deltaTime / 120);
        vec3.add(cameraPos, cameraPos, camDisp);
    }
}

function render(time) {
    if (program === null) return;

    currentTime = time - startTime;
    deltaTime = time - lastTime;

    moveCamera();

    // Tell WebGL to use our shader program.
    glContext.useProgram(program);

    glContext.clearColor(0.0, 0.0, 0.0, 1.0);
    glContext.clear(glContext.COLOR_BUFFER_BIT);

    setViewToWorld();

    writeToFFTTexture();

    // Set uniforms.
    glContext.uniform3f(cameraPosUniform, cameraPos[0], cameraPos[1], cameraPos[2]);
    glContext.uniformMatrix4fv(viewToWorldUniform, false, viewToWorldMat);
    glContext.uniform2f(screenSizeUniform, glCanvas.width, glCanvas.height);
    glContext.uniform1f(timeUniform, currentTime / 1000);

    glContext.activeTexture(glContext.TEXTURE0);
    glContext.bindTexture(glContext.TEXTURE_2D, frequencyTexture);
    glContext.uniform1i(frequencyDataUniform, 0);

    // Actual drawing
    glContext.drawArrays(glContext.TRIANGLES, 0, 6);
	glContext.finish();

	// Now we do FPS calculation.
	var fps = 1000/deltaTime;
	fpsElement.innerHTML = "FPS: "+fps.toFixed(2);

	lastTime = time;

    nextRenderFrame = requestAnimationFrame(render, glCanvas);
}

function recompileShader() {
    // Note: This won't happen because we load the
    // source before we initialise, unless the user clicks
    // the button somehow while the page is loading,
    // in which case we should do nothing anyway.
	if (headerCodeMirror === null || fragCodeMirror === null) return;

    // Stop the render loop.
    if (nextRenderFrame !== null) cancelAnimationFrame(nextRenderFrame);

    fpsElement.innerHTML = '';

    var fragSrc = headerCodeMirror.getValue() + "\n" + fragCodeMirror.getValue();

    program = generateProgramFromSources(vertexSrc, fragSrc);

    if (program === null) return;

    getProgramAttribLocations();

    resetCamera();

    // We now restart the render loop.
    startTime = performance.now();
    lastTime = startTime;
    nextRenderFrame = requestAnimationFrame(render, glCanvas);
}
