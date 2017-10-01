// Header file for the fragment shader.
// Prepended to the file.
// Defines uniforms.
// Read only.

// This sets the float precision to medium,
// for best compatibility with mobile devices.
precision mediump float;


// This contains the position of the camera,
// as a vector containing three floats: the
// x, y and z position of the camera.
uniform vec3 cameraPos;

// This is a matrix representing a rotation
// from a ray being emitted by the camera
// to a 'world' ray, and is calculated by the
// JavaScript code based on the mouse position.
// Technically only the top-left 3x3 matrix
// of this matrix is relevant, since that is the
// component which contains rotation data;
// however, this is a 4x4 matrix for compatibility
// with other 3D transformation matrices, which
// are often 4x4.
// Use: vec3 new = (viewToWorld * vec4(old, 1.0)).xyz;
uniform mat4 viewToWorld;

// This is a matrix containing the screen size.
// The width is the x component, and the height
// is the y component.
// This can be used in conjunction with gl_FragCoord
// to calculate the current fragment's "relative"
// location on the screen.
uniform vec2 screenSize;

// This stores the current time, i.e. the number
// of seconds since the program began execution.
uniform float time;

// This stores the frequency of the audio
// input, as a 512x1 texture.
uniform sampler2D frequencyData;
