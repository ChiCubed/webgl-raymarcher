// Fragment shader source.

// This is a basic raymarching template, with
// a simple scene. To play around without
// getting your hands dirty, mess with the
// 'scene' function.

// Some raymarching constants.
const int MAX_MARCH_STEPS = 128;
const int MAX_SHADOW_MARCH_STEPS = 64;
const float NEAR_DIST = 0.01;
const float FAR_DIST = 256.0;
float FOV = 45.0;
const float EPSILON = 0.001;
const float NORMAL_EPSILON = 0.01;
const float stepScale = 0.90;

// For material lightings.
vec3 AMBIENT_COLOUR = vec3(0.0);
float ambientIntensity = 1.0;

struct HitPoint {
    float dist; // distance to scene
    vec3 diffuse; // diffuse colour
};

// Helper functions for HitPoint
HitPoint min(HitPoint a, HitPoint b) {
    if (a.dist < b.dist) {
        return a;
    } else {
        return b;
    }
}

HitPoint max(HitPoint a, HitPoint b) {
    if (a.dist > b.dist) {
        return a;
    } else {
        return b;
    }
}

// Helper functions for transposition.
// https://github.com/glslify/glsl-transpose/blob/master/index.glsl
float transpose(float m) {
    return m;
}

mat2 transpose(mat2 m) {
    return mat2(m[0][0], m[1][0],
                m[0][1], m[1][1]);
}

mat3 transpose(mat3 m) {
    return mat3(m[0][0], m[1][0], m[2][0],
                m[0][1], m[1][1], m[2][1],
                m[0][2], m[1][2], m[2][2]);
}

mat4 transpose(mat4 m) {
    return mat4(m[0][0], m[1][0], m[2][0], m[3][0],
                m[0][1], m[1][1], m[2][1], m[3][1],
                m[0][2], m[1][2], m[2][2], m[3][2],
                m[0][3], m[1][3], m[2][3], m[3][3]);
}


// Smooth minimum.
// Can be used to 'blend' objects.
// http://www.iquilezles.org/www/articles/smin/smin.htm
// The result of smin on objects may not result
// in a true distance function, since the new
// distance function may overestimate some distances.
// Thus the step size must be downscaled.
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b,a,h) - k*h*(1.0-h);
}

// Distance functions.
// Basically a word-for-word translation of:
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float udBox(vec3 p, vec3 b) {
    return length(max(abs(p)-b,0.0));
}

float udRoundBox(vec3 p, vec3 b, float r) {
    return length(max(abs(p)-b,0.0))-r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float sdCylinder(vec3 p, vec3 c) {
    return length(p.xz-c.xy)-c.z;
}

float sdCone(vec3 p, vec2 c) {
    // c must be normalized
    // i.e. length(c) = 1.0
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}

float sdPlane(vec3 p, vec4 n) {
    // n must be normalized
    return dot(p,n.xyz) + n.w;
}

float sdHexPrism(vec3 p, vec2 h) {
    vec3 q = abs(p);
    return max(q.z-h.y,max((q.x*0.866025+q.y*0.5),q.y)-h.x);
}

float sdTriPrism(vec3 p, vec2 h) {
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

float sdCappedCylinder(vec3 p, vec2 h) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCappedCone(vec3 p, vec3 c) {
    // This function is 'bound'
    // i.e. not an overestimate.
    vec2 q  = vec2(length(p.xz),p.y);
    vec2 v  = vec2(c.z*c.y/c.x, -c.z);
    vec2 w  = v-q;
    vec2 vv = vec2(dot(v,v), v.x*v.x);
    vec2 qv = vec2(dot(v,w), v.x*w.x);
    vec2 d  = max(qv,0.0)*qv/vv;
    return sqrt(dot(w,w) - max(d.x,d.y)) * sign(max(q.y*v.x-q.x*v.y,w.y));
}

float sdEllipsoid(vec3 p, vec3 r) {
    // This function is also 'bound'.
    return (length(p/r) - 1.0)*min(min(r.x,r.y),r.z);
}

float dot2(vec3 v) { return dot(v,v); }

float udTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ba = b-a; vec3 pa = p-a;
    vec3 cb = c-b; vec3 pb = p-b;
    vec3 ac = a-c; vec3 pc = p-c;
    vec3 nor = cross(ba,ac);

    return sqrt(
        (sign(dot(cross(ba,nor),pa)) +
         sign(dot(cross(cb,nor),pb)) +
         sign(dot(cross(ac,nor),pc)) < 2.0)
        ?
        min(min(
            dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
            dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb)),
            dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc))
        :
        dot(nor,pa)*dot(nor,pa)/dot2(nor)
    );
}

float udQuad(vec3 p, vec3 a, vec3 b, vec3 c, vec3 d) {
    vec3 ba = b-a; vec3 pa = p-a;
    vec3 cb = c-b; vec3 pb = p-b;
    vec3 dc = d-c; vec3 pc = p-c;
    vec3 ad = a-d; vec3 pd = p-d;
    vec3 nor = cross(ba,ad);

    return sqrt(
        (sign(dot(cross(ba,nor),pa)) +
         sign(dot(cross(cb,nor),pb)) +
         sign(dot(cross(dc,nor),pc)) +
         sign(dot(cross(ad,nor),pd)) < 3.0)
        ?
        min(min(min(
            dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
            dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb)),
            dot2(dc*clamp(dot(dc,pc)/dot2(dc),0.0,1.0)-pc)),
            dot2(ad*clamp(dot(ad,pd)/dot2(ad),0.0,1.0)-pd))
        :
        dot(nor,pa)*dot(nor,pa)/dot2(nor)
    );
}


float opU(float d1, float d2) {
    return min(d1,d2);
}

float opS(float d1, float d2) {
    return max(-d1,d2);
}

float opI(float d1, float d2) {
    return max(d1,d2);
}

vec3 opRep(vec3 p, vec3 c) {
    // usage: distFunc(opRep(p,c))
    return mod(p,c)-0.5*c;
}

mat4 invertTrans(mat4 m) {
    // Calculates the inverse of a
    // mat4, given that it is a
    // transformation matrix.
    mat3 R = transpose(mat3(m));
    vec3 t = -R*vec3(m[3][0],m[3][1],m[3][2]);
    return mat4(R[0][0], R[1][0], R[2][0], t[0],
                R[0][1], R[1][1], R[2][1], t[1],
                R[0][2], R[1][2], R[2][2], t[2],
                      0,       0,       0,    1);
}

vec3 opTrans(vec3 p, mat4 m) {
    // usage: distFunc(opRep(p,c))
    return vec3(invertTrans(m)*vec4(p,1.0));
}

// Distance function for the scene
HitPoint scene(vec3 p) {
    return min(HitPoint(sdSphere(p,1.0),vec3(1,0,0)),
               HitPoint(sdSphere(p-vec3(1,0,2.0*sin(time)),1.0),vec3(0,0,1)));
}

// Estimate normal at a point
// for lighting calcs
vec3 estimateNormal(vec3 p) {
    vec2 eps = vec2(NORMAL_EPSILON, 0.0);
    return normalize(vec3(
        scene(p+eps.xyy).dist - scene(p-eps.xyy).dist,
        scene(p+eps.yxy).dist - scene(p-eps.yxy).dist,
        scene(p+eps.yyx).dist - scene(p-eps.yyx).dist)
    );
}

HitPoint march(vec3 ro, vec3 rd, float near, float far) {
    // ro and rd are ray origin and ray direction respectively
    float depth = near;
    HitPoint c;
    for (int i=0; i<MAX_MARCH_STEPS; ++i) {
        c = scene(ro + depth * rd);
        if (abs(c.dist) < EPSILON) break;
        depth += c.dist * stepScale;
        if (depth >= far) break;
    }
    c.dist = depth;
    return c;
}

// Calculates the amount that a pixel
// is in shadow from a light.
// Based on http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float shadow(vec3 ro, vec3 rd, float near, float far, float k) {
	float res = 1.0, dist, t = near;
	for (int i = 0; i < MAX_SHADOW_MARCH_STEPS; ++i) {
		dist = scene(ro + rd*t).dist;
        if (dist < EPSILON) return 0.0;
        if (t >= far) break;
		res = min(res, k*dist/t);
        t += dist;
	}

	return clamp(res, 0.0, 1.0);
}

// Phong lighting of a point.
// Includes attenuation i.e. lights
// are weaker from father away.
// See Wikipedia for an explanation
// of how this algorithm works.
// recLightRangeSqr is
// used for attenuation calculation.
vec3 phongLighting(vec3 diffuse_col, vec3 specular_col, float alpha,
                   vec3 p, vec3 normal, vec3 cam, vec3 viewerNormal,
                   vec3 lightPos, vec3 lightColour, float intensity,
                   float recLightRangeSqr) {
	// The normal at the point
    vec3 N = normal;
	// This is used for attenuation
	// calculation.
	vec3 relativePos = lightPos - p;
	vec3 L = normalize(relativePos);
	vec3 V = viewerNormal;
	vec3 R = normalize(reflect(-L, N));

	float dotLN = dot(L, N);
	float dotRV = dot(R, V);

	if (dotLN < 0.0) {
		// The surface is facing away
		// from the light i.e.
		// receives no light.
		// Since this function returns
		// the colour, obviously
		// this pixel is completely black.
        // (or the contribution from this
        //  light, at least, is nothing.)
		return vec3(0.0);
	}

	float squareDist = dot(relativePos, relativePos);
	float attenuatedIntensity = 1.0 - squareDist * recLightRangeSqr;

	if (attenuatedIntensity < 0.0) {
		// The point is clearly completely
		// unlit, since it is outside
		// the light's range.
		return vec3(0.0);
	}

	// Now square and multiply by
	// the light's actual intensity.
	attenuatedIntensity *= attenuatedIntensity * intensity;

	if (dotRV < 0.0) {
		// The surface has no specular lighting
		return (diffuse_col*dotLN)*lightColour * attenuatedIntensity;
	}

	// Approximate the specular component
	const int gamma = 8;
	const int gamma_log2 = 3;
	float calculated_specular = (1.0 - alpha * (1.0-dotRV)/float(gamma));
	for (int i=0; i<gamma_log2; ++i) {
		calculated_specular *= calculated_specular;
	}

	return (diffuse_col*dotLN + specular_col*calculated_specular)*lightColour * attenuatedIntensity;
}

// Phong lighting, but for directional lights.
vec3 directionalPhongLighting(vec3 diffuse_col, vec3 specular_col, float alpha,
                              vec3 p, vec3 normal, vec3 cam, vec3 viewerNormal,
                              vec3 lightDirection, vec3 lightColour, float intensity) {
    vec3 N = normal;
	vec3 L = -lightDirection;
	vec3 V = viewerNormal;
	vec3 R = normalize(reflect(-L, N));

	float dotLN = dot(L, N);
	float dotRV = dot(R, V);

	if (dotLN < 0.0) return vec3(0.0);
	// Attenuation doesn't make sense
	// with directional lights, so
	// we omit it.
	if (dotRV < 0.0) return (diffuse_col*dotLN)*lightColour*intensity;

	const int gamma = 8;
	const int gamma_log2 = 3;
	float calculated_specular = (1.0 - alpha * (1.0-dotRV)/float(gamma));
	for (int i=0; i<gamma_log2; ++i) {
		calculated_specular *= calculated_specular;
	}

	return (diffuse_col*dotLN + specular_col*calculated_specular)*lightColour*intensity;
}

// This function calculates the colour of a pixel according to
// the lighting and its diffuse colour.
vec3 lighting(vec3 ambient_col, vec3 diffuse_col, vec3 specular_col,
              float alpha, vec3 p, vec3 cam, float ambientIntensity) {
    // current colour
    vec3 colour = ambient_col * ambientIntensity;

    // normals
    vec3 normal = estimateNormal(p);
    vec3 viewerNormal = normalize(cam - p);

    // point lighting
    vec3 lightPos = vec3(0,0,-10);

    vec3 tmp = phongLighting(diffuse_col, specular_col, alpha, p, normal, cam,
                             viewerNormal, lightPos, vec3(1,1,1), 1.0, 0.0025);

    // if the point is lit at all:
    if (tmp != vec3(0.0)) {
        // calculate shadowing
        vec3 relPos = lightPos - p;
        float dist = length(relPos);
        vec3 relDir = relPos / dist;
        // The reason we add norm*EPSILON*2
        // is to move the start point for the
        // shadowing slightly away from the
        // point which we originally intersected,
        // to prevent artefacts when the normal is
        // almost perpendicular to the light.
        tmp *= shadow(p + normal*EPSILON*2.0, relDir, EPSILON*2.0, dist, 8.0);
    }

    colour += tmp;

    // directional lighting
    // vec3 tmp = directionalPhongLighting(diffuse_col, specular_col, alpha, p, normal, cam, viewerNormal, DIRECTION, COLOUR, INTENSITY);
    //
    // if (tmp != vec3(0.0)) {
    //     // We arbitrarily set the
    //     // 'far' plane to 100,
    //     // so if an object is not
    //     // in shadow within 100
    //     // units it is not
    //     // in shadow at all.
    //     tmp *= shadow(p - DIRECTION*EPSILON*2.0,
    //                   -DIRECTION, EPSILON*2.0, 100.0, 8.0);
    // }
    //
    // colour += tmp;

	return colour;
}


// This function centers coordinates on
// the screen, so instead of going
// from 0 -> screen size in the x and y directions
// the coordinates go from
// -screen size / 2 -> screen size / 2.
vec2 centerCoordinates(vec2 coord, vec2 size) {
    return coord - 0.5*size;
}


// This function calculates the direction
// of the ray given the:
//   Field of View of the camera
//   The screen-space coordinates of the pixel
//   The size of the screen
vec3 direction(float fov, vec2 coord, vec2 size) {
    vec2 xy = centerCoordinates(coord, size);
    // Calculate the z component based on angle
    float z = size.y / tan(radians(fov) * 0.5);

    vec3 tmp = vec3(xy, -z);
    return normalize(tmp);
}

void main() {
    // Calculate ray direction.
    // gl_FragCoord.xy is the position of the
    // pixel in screen space.
    vec3 rayDir = direction(FOV, gl_FragCoord.xy, screenSize);

    // Convert the ray direction to world coordinates.
    vec3 worldRayDir = vec3(viewToWorld * vec4(rayDir,1));

    // Cast a ray
    HitPoint result = march(cameraPos, worldRayDir, NEAR_DIST, FAR_DIST);

    if (result.dist < FAR_DIST - EPSILON) {
        // Calculate colour based on lights
        vec3 colour = lighting(AMBIENT_COLOUR, result.diffuse, vec3(1,1,1),
                               4.0, cameraPos + worldRayDir*result.dist, cameraPos, ambientIntensity);
        gl_FragColor = vec4(colour,1);
    } else {
        gl_FragColor = vec4(0,0,0,1);
    }
}
