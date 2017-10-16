// Fragment shader source.

// This is a simple scene using
// a combination of raytracing
// and raymarching to make some sort
// of 3D spectrogram.
// The code was written from scratch,
// but I have used
// https://www.shadertoy.com/view/Msl3Rr
// (which is extremely similar to this scene)
// as a guideline while writing this.

precision mediump float;

// Some raymarching constants.
const int MAX_TRACE_STEPS = 128;
const int MAX_MARCH_STEPS = 32;
const int MAX_SHADOW_MARCH_STEPS = 32;
const float NEAR_DIST = 0.001;
const float FAR_DIST = 80.0;
float FOV = 60.0;
const float EPSILON = 0.001;
const float NORMAL_EPSILON = 0.001;
const vec3 eps = vec3(1.0, -1.0, 0.0) * NORMAL_EPSILON;
const float stepScale = 0.90;
const int MAX_REFLECTIONS = 3;

#define CHEAP_NORMALS
#define ENABLE_CLOUDS

const float M_PI = 3.14159265358979323846;

// For material lightings.
vec3 AMBIENT_COLOUR = vec3(0.05);
float ambientIntensity = 1.0;

struct HitPoint {
    float dist; // distance to scene
    vec3 diffuse; // diffuse colour
	float reflectivity;
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

// Perlin noise.
// https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
// As far as I can tell, this version is a derivative of
// https://github.com/stegu/webgl-noise/blob/master/src/classicnoise3D.glsl
vec4 permute(vec4 x) {return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r) {return 1.79284291400159 - 0.85373472095314 * r;}
vec3 fade(vec3 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

float cnoise(vec3 P) {
    vec3 Pi0 = floor(P); // Integer part for indexing
    vec3 Pi1 = Pi0 + vec3(1.0); // Integer part + 1
    Pi0 = mod(Pi0, 289.0);
    Pi1 = mod(Pi1, 289.0);
    vec3 Pf0 = fract(P); // Fractional part for interpolation
    vec3 Pf1 = Pf0 - vec3(1.0); // Fractional part - 1.0
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);

    vec4 gx0 = ixy0 / 7.0;
    vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);

    vec4 gx1 = ixy1 / 7.0;
    vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);

    vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
    vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
    vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
    vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
    vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);

    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

// Simplex noise.
// https://github.com/stegu/webgl-noise/blob/master/src/noise3D.glsl

vec3 mod289(vec3 x) {return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x) {return x-floor(x*(1.0/289.0))*289.0;}

float snoise(vec3 v) {
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute( permute( permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                  dot(p2,x2), dot(p3,x3) ) );
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

float smax(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(a,b,h) + k*h*(1.0-h);
}

// Smooth minimum for HitPoints.
// Attempts to do some sort of colour blending...
HitPoint smin(HitPoint a, HitPoint b, float k) {
    float h = clamp(0.5 + 0.5*(b.dist-a.dist)/k, 0.0, 1.0);
    return HitPoint(mix(b.dist,a.dist,h)-k*h*(1.0-h),
                    mix(b.diffuse,a.diffuse,smoothstep(0.0,1.0,h)),mix(b.reflectivity,a.reflectivity,smoothstep(0.0,1.0,h)));
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
    return mat4(R[0][0], R[0][1], R[0][2], t[0],
                R[1][0], R[1][1], R[1][2], t[1],
                R[2][0], R[2][1], R[2][2], t[2],
                      0,       0,       0,    1);
}

vec3 opTrans(vec3 p, mat4 m) {
    // usage: distFunc(opRep(p,c))
    return vec3(invertTrans(m)*vec4(p,1.0));
}

mat3 rotateX(float x) {
    float c = cos(x);
    float s = sin(x);
    return mat3(
        1.0, 0.0, 0.0,
        0.0,   c,   s,
        0.0,  -s,   c
    );
}

mat3 rotateY(float x) {
    float c = cos(x);
    float s = sin(x);
    return mat3(
          c, 0.0,  -s,
        0.0, 1.0, 0.0,
          s, 0.0,   c
    );
}

float hash(float n) {
	return fract(sin(n)*13.5453123);
}

// http://iquilezles.org/www/articles/functions/functions.htm
float cubicPulse(float w, float x) {
    x = abs(x);
    if (x>w) return 0.0;
    x /= w;
    return 1.0 - x*x*(3.0-2.0*x);
}

// Get the coordinate of the texture
// at point p, given it's been floored.
float texCoord(vec2 ip) {
    return hash(dot(vec2(50.8,1.3),ip));
}

// Get height of rect
// at position p.
float height(vec2 p) {
    vec2 ip = floor(p);
    float r = texCoord(ip);
    return texture2D(frequencyData, vec2(r, 0.5)).x * 12.0
        * (cubicPulse(40.0, length(ip)) + 0.1) // This makes bars closer to the center higher.
        * (r*0.65 + 0.35) // This makes bars with higher frequencies higher.
        ;
}

// Distance function for the scene
HitPoint scene(vec3 p) {
    HitPoint res = HitPoint(FAR_DIST, vec3(0.0), 0.0);

    vec2 m = fract(p.xz);
	vec2 ip = floor(p.xz);
    float h = height(p.xz);

	float box = udRoundBox(vec3(m.x-0.5,p.y-h*0.5,m.y-0.5), vec3(0.4,h*0.5,0.4), 0.1);

    float r = texCoord(ip);
    res = min(res, HitPoint(box, vec3(1),
                            floor(r+0.3)) // Reflectivity is determined by frequency
             );

    return res;
}

// Estimate normal at a point
// for lighting calcs
vec3 estimateNormal(vec3 p) {
	#ifdef CHEAP_NORMALS
		return normalize(
			eps.xyy * scene(p+eps.xyy).dist +
			eps.yyx * scene(p+eps.yyx).dist +
			eps.yxy * scene(p+eps.yxy).dist +
			eps.xxx * scene(p+eps.xxx).dist
		);
	#else
		return normalize(vec3(
			scene(p+eps.xzz).dist - scene(p-eps.xzz).dist,
			scene(p+eps.zxz).dist - scene(p-eps.zxz).dist,
			scene(p+eps.zzx).dist - scene(p-eps.zzx).dist)
		);
    #endif
}

// https://tavianator.com/fast-branchless-raybounding-box-intersections/
vec2 intersection(float miny, float maxy, vec3 ro, vec3 rid) {
    vec2 b = floor(ro.xz);

    float t1 = (miny    - ro.y)*rid.y;
    float t2 = (maxy    - ro.y)*rid.y;

    float tmin = min(t1, t2);
    float tmax = max(t1, t2);

    return tmax > max(tmin, 0.0) ? vec2(max(tmin,0.0),tmax) : vec2(-1.0);
}

float infintersection(vec3 ro, vec3 rid) {
    vec2 b = floor(ro.xz);

    float t1 = (b.x     - ro.x)*rid.x;
    float t2 = (b.x+1.0 - ro.x)*rid.x;

    float tmin = min(t1, t2);
    float tmax = max(t1, t2);

    t1 = (b.y     - ro.z)*rid.z;
    t2 = (b.y+1.0 - ro.z)*rid.z;

    tmin = max(tmin, min(min(t1, t2), tmax));
    tmax = min(tmax, max(max(t1, t2), tmin));

    return tmax;
}

HitPoint march(vec3 ro, vec3 rd, float near, float far, int numTraceSteps) {
    // ro and rd are ray origin and ray direction respectively
    float depth = near, t, h;
	vec2 inter;
    HitPoint c;

    vec2 s;
	vec3 rid = 1.0 / rd;
    vec3 p = ro + depth * rd;

    for (int i=0; i<MAX_TRACE_STEPS; ++i) {
		if (i >= numTraceSteps) break;
        // Raymarch across a 2D grid
        // until we hit a block.
        h = height(p.xz);
        // Check box intersection here
		float infinter = infintersection(p,rid)+depth;
        vec2 inter = intersection(-0.1,h+0.1,p,rid)+depth;
		inter.y = min(inter.y, infinter);
        if (inter.x >= depth) {
            // Time to actually raymarch
            depth = inter.x;
			p = ro + depth * rd;

            for (int j=0; j<MAX_MARCH_STEPS; ++j) {
                c = scene(p);
                if (abs(c.dist) < EPSILON) break;

                depth += c.dist * stepScale;
				p = ro + depth * rd;
                if (depth > inter.y || depth >= far) break;
            }
			if (depth <= inter.y) {
            	c.dist = depth;
            	return c;
			}
		}
		depth = infinter+EPSILON;
		p = ro + depth * rd;

		if (depth >= far) break;
    }
    c.dist = far;
    return c;
}

// Calculates the amount that a pixel
// is in shadow from a light.
// Based on http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
float shadow(vec3 ro, vec3 rd, float near, float far, float k, int numMarchSteps) {
	float res = 1.0, dist, t = near;
	vec3 p = ro + rd*t;
	vec3 rid = 1.0 / rd;
	for (int i = 0; i < MAX_SHADOW_MARCH_STEPS; ++i) {
		if (i >= numMarchSteps) break;

		dist = scene(p).dist;
		res = min(res, k*dist/t);

		// go to next cell
		// or just step
        t += min(dist * stepScale, infintersection(p, rid)+EPSILON);
		p = ro + rd*t;

		if (dist < EPSILON || t >= far) break;
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


vec3 sunDir; // A global variable for the sun light's direction.
vec3 lightPos, lightCol;


// This function calculates the colour of a pixel according to
// the lighting and its diffuse colour.
vec3 lighting(vec3 ambient_col, vec3 diffuse_col, vec3 specular_col,
              float alpha, vec3 normal, vec3 p, vec3 cam, float ambientIntensity, int numShadowMarchSteps) {
    // current colour
    vec3 colour = ambient_col * ambientIntensity;

    // normals
    vec3 viewerNormal = normalize(cam - p);

    // point lighting
    vec3 tmp = phongLighting(diffuse_col, specular_col, alpha, p, normal, cam,
                             viewerNormal, lightPos, lightCol, 0.7, 0.0001);

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
        tmp *= shadow(p + normal*EPSILON*2.0, relDir, EPSILON*2.0, dist, 8.0, numShadowMarchSteps);
    }

	colour += tmp;

    // directional lighting: sun
	if (sunDir.y < 0.0) {
		tmp = directionalPhongLighting(diffuse_col, specular_col, alpha, p, normal, cam,
									   viewerNormal, sunDir, vec3(1, 0.8, 0), 0.4);

		// We arbitrarily set the
		// 'far' plane to FAR_DIST,
		// so if an object is not
		// in shadow within FAR_DIST
		// units it is not
		// in shadow at all.
		tmp *= shadow(p + normal*EPSILON*2.0,
					  -sunDir, EPSILON*2.0, FAR_DIST, 8.0, numShadowMarchSteps);

		colour += tmp;
	} else {
		// moon:
		// direction is the negation of sunDir
		tmp = directionalPhongLighting(diffuse_col, specular_col, alpha, p, normal, cam,
									   viewerNormal, -sunDir, vec3(0.9, 0.9, 1), 0.25);

		tmp *= shadow(p + normal*EPSILON*2.0,
					  sunDir, EPSILON*2.0, FAR_DIST, 8.0, numShadowMarchSteps);

		colour += tmp;
	}

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

vec3 render(vec3 ro, vec3 rd) {
	HitPoint result;
	vec3 colour, tmpColour, skyColour;
	float alpha = 1.0, fogAmt;
	for (int i = 0; i < MAX_REFLECTIONS + 1; i++) {
		result = march(ro, rd, NEAR_DIST, FAR_DIST, 128 - i*48);

		// Set the sky colour
		skyColour = clamp(vec3(0.1,0.6,1.0)*(rd.y*0.2+0.8)*(-sunDir.y), vec3(0.0), vec3(1.0));
		// sun
		float sun = clamp(1.0 - length(-sunDir-rd), 0.0, 1.0);
		skyColour += vec3(0.8, 0.5, 0.0) * sun;

		// moon
		float moonIntensity = 1.0 - length(sunDir-rd);
		moonIntensity = pow(moonIntensity, 5.0);
		skyColour += vec3(0.9, 0.9, 1.0) * clamp(moonIntensity, 0.0, 1.0);

		skyColour = clamp(skyColour,vec3(0),vec3(1));

		// clouds
		// https://www.shadertoy.com/view/MdBGzG
		#ifdef ENABLE_CLOUDS
		float pt = (1000.0-ro.y)/rd.y;
		if (pt > 0.0) {
			vec3 spos = ro + pt*rd;
            vec3 fpos = vec3(spos.xz*0.0001-vec2(time*0.02,0), time*0.01);
			float clo = clamp(1.0
                - snoise(fpos*4.0) - 0.3
                + snoise(fpos*24.0)*0.3 - 0.03
                - snoise(fpos*64.0)*0.1 + 0.01,
                0.0, 1.0);
			vec3 cloCol = mix(vec3(0.4,0.5,0.6), vec3(1.3,0.6,0.4), pow(sun,2.0))*(0.5+0.5*clo);
			skyColour = mix(skyColour, cloCol, 0.5*smoothstep(0.4, 1.0, clo));
		}
		#endif

		if (result.dist < FAR_DIST - EPSILON) {
			// Calculate colour based on lights
			vec3 normal = estimateNormal(ro + result.dist*rd);
			tmpColour = lighting(AMBIENT_COLOUR, result.diffuse, vec3(1,1,1),
								   4.0, normal, ro + result.dist*rd, cameraPos, ambientIntensity, MAX_SHADOW_MARCH_STEPS - 16*i);

			// Apply fog
			fogAmt = pow(result.dist / FAR_DIST, 1.5);
			tmpColour = mix(tmpColour, skyColour, fogAmt);

			// Handle reflection
			colour += tmpColour * alpha;

			// Second part of fog
			alpha *= 1.0 - fogAmt;

			if (result.reflectivity == 0.0) break;

			ro += rd * (result.dist - EPSILON);
			rd = reflect(rd, normal);
			alpha *= result.reflectivity;
		} else {
			tmpColour = skyColour;

			colour += tmpColour * alpha;
			break;
		}
	}

	return colour;
}

void main() {
    sunDir = rotateX(time*0.1)*vec3(0.0, -1.0, 0.0);
	lightPos = 15.0*vec3(sin(time),1,cos(time));
	lightCol = vec3(sin(time*1.2+0.4)*0.2+0.7,cos(time*1.5+4.2)*0.3+0.5,sin(time*0.3+0.2)*0.3+0.4);


    // For fun, let's make the FOV pulsate
    // based on the amount of bass.
    float bass = texture2D(frequencyData, vec2(0.1, 0.5)).x;

    // Calculate ray direction.
    // gl_FragCoord.xy is the position of the
    // pixel in screen space.
    vec3 rayDir = direction(FOV - bass*10.0, gl_FragCoord.xy, screenSize);

    // Convert the ray direction to world coordinates.
    vec3 worldRayDir = vec3(viewToWorld * vec4(rayDir,1));

	vec3 ro = cameraPos;

    // Render
	vec3 colour = render(ro, worldRayDir);

    // Gamma correction
    colour = pow(colour, vec3(0.85));

    // Vignetting
    vec2 uv = gl_FragCoord.xy / screenSize;
    uv *= 1.0 - uv;
    float vig = pow(uv.x*uv.y * 16.0, 0.3);

	gl_FragColor = vec4(colour * vig, 1);
}
