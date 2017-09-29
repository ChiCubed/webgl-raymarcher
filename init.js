var headerCodeMirror = null, fragCodeMirror = null;
var glContext, glCanvas;

// yaw, pitch, roll
var angle = [0.0, 0.0, 0.0];
var viewToWorldMat;

function init() {
	glCanvas = document.getElementById('gl-canvas');

	// Get WebGL context.
	glContext = glCanvas.getContext('webgl') || glCanvas.getContext('experimental-webgl');

	if (!glContext) {
		alert('Failed to initialise WebGL.');
	}
}

function setViewToWorld() {
	mat4.identity(viewToWorldMat);
	mat4.rotateY(viewToWorldMat, viewToWorldMat, angle[0]);
	mat4.rotateX(viewToWorldMat, viewToWorldMat, angle[1]);
	mat4.rotateZ(viewToWorldMat, viewToWorldMat, angle[2]);
}

function compileFragShader() {
	if (headerCodeMirror === null || fragCodeMirror === null) return;
}
