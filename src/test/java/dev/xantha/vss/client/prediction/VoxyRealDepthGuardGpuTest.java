package dev.xantha.vss.client.prediction;

import static org.junit.jupiter.api.Assertions.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL45C.*;

import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import dev.xantha.vss.compat.VoxyRealDepthGuard;
import java.nio.charset.StandardCharsets;
import java.util.regex.Pattern;
import java.util.zip.ZipFile;
import org.joml.Matrix4f;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL;

/** Runs the installed Voxy final blit, with every VSS patch, over real and empty destination depth. */
@EnabledIfSystemProperty(named = "vss.gpuTests", matches = "true")
@EnabledIfSystemProperty(named = "vss.voxyJar", matches = ".+")
class VoxyRealDepthGuardGpuTest {
    private static final String BLIT = "voxy:post/blit_texture_depth_cutout.frag";
    private static final float[] REAL = {.9f, .1f, .1f};
    private static final float[] LOD = {.1f, .9f, .1f};

    @Test void lodDepthCannotBuryRealGeometryItIsNotClearlyInFrontOf() throws Exception {
        assertTrue(glfwInit());
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        long window = glfwCreateWindow(16, 16, "VSS real depth guard", 0, 0);
        assertNotEquals(0, window);
        try (var jar = new ZipFile(System.getProperty("vss.voxyJar"))) {
            glfwMakeContextCurrent(window); GL.createCapabilities(); RenderSystem.initRenderThread();
            String fragment = load(jar, BLIT).replaceFirst("\n", "\n#define EMIT_COLOUR\n");
            fragment = PredictionFogBridge.patch(BLIT, fragment);
            fragment = PredictionVoxyBoundaryBridge.patch(BLIT, fragment);
            fragment = VoxyRealDepthGuard.patch(BLIT, fragment);
            assertTrue(fragment.contains("VssRealDepthGuard"), "installed final blit must accept the guard patch");
            var main = new TextureTarget(16, 16, true, false);
            var voxy = new TextureTarget(16, 16, true, false);
            int vao = glGenVertexArrays();
            try (var blit = GlProgram.link("guarded_voxy_final_blit", """
                    #version 450 core
                    out vec2 UV;
                    void main() {
                        vec2 p = vec2(gl_VertexID == 1 ? 3 : -1, gl_VertexID == 2 ? 3 : -1);
                        UV = p * .5 + .5; gl_Position = vec4(p, 0, 1);
                    }
                    """, fragment);
                 var plane = GlProgram.link("guard_plane", """
                    #version 450 core
                    uniform mat4 Projection;
                    uniform float Distance;
                    void main() {
                        vec2 p = vec2(gl_VertexID == 1 ? 3 : -1, gl_VertexID == 2 ? 3 : -1);
                        vec4 center = Projection * vec4(0, 0, -Distance, 1);
                        gl_Position = vec4(p * center.w, center.zw);
                    }
                    """, """
                    #version 450 core
                    uniform vec4 SurfaceColor;
                    out vec4 colour;
                    void main() { colour = SurfaceColor; }
                    """)) {
                var vanilla = new Matrix4f().perspective(1.2F, 1, .05F, 2048);
                var lod = new Matrix4f().perspective(1.2F, 1, 16, 48000);
                Case[] cases = {
                        // Sky: VSS's forced LOD depth write is retained.
                        new Case(Float.NaN, 100F, true, true),
                        // Stale LOD a few centimetres in front of a real face (the drawer case).
                        new Case(100F, 99.95F, true, false),
                        // Coplanar LOD just behind: real geometry wins.
                        new Case(100F, 100.2F, true, false),
                        // A LOD hill clearly in front of distant real geometry still occludes it.
                        new Case(500F, 300F, true, true),
                        // Control: without the guard the stale LOD buries the real face.
                        new Case(100F, 99.95F, false, true),
                };
                for (Case c : cases) {
                    glBindVertexArray(vao);
                    glDisable(GL_BLEND); glDisable(GL_CULL_FACE); glEnable(GL_DEPTH_TEST);
                    glDepthFunc(GL_ALWAYS); glDepthMask(true);
                    voxy.bindWrite(true); glClearDepth(1); glClearColor(0, 0, 0, 0);
                    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
                    draw(plane, lod, c.lod(), LOD);
                    main.bindWrite(true); glClearColor(0, 0, 0, 1);
                    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
                    if (!Float.isNaN(c.real())) draw(plane, vanilla, c.real(), REAL);

                    // Live order: snapshot at finish HEAD, uniforms right before the blit, restore at RETURN.
                    if (c.guard()) VoxyRealDepthGuard.prepare(main.frameBufferId);
                    main.bindWrite(true); glBindVertexArray(vao);
                    blit.use();
                    glBindTextureUnit(0, voxy.getDepthTextureId()); glBindTextureUnit(3, voxy.getColorTextureId());
                    matrix(1, new Matrix4f(lod).invert()); matrix(2, vanilla);
                    VoxyRealDepthGuard.bind();
                    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL); glDepthMask(true);
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                    VoxyRealDepthGuard.restore();

                    var depth = BufferUtils.createFloatBuffer(1);
                    glReadPixels(8, 8, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
                    var pixel = BufferUtils.createByteBuffer(4);
                    glReadPixels(8, 8, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
                    float expectedDistance = c.lodWins() ? c.lod() : c.real();
                    float expected = window(vanilla, expectedDistance);
                    assertEquals(expected, depth.get(0), 4F / 16777215F, c + ": destination depth");
                    float[] colour = c.lodWins() ? LOD : REAL;
                    for (int i = 0; i < 3; i++)
                        assertEquals(Math.round(colour[i] * 255), pixel.get(i) & 255, 2, c + ": colour channel " + i);
                    assertEquals(GL_NO_ERROR, glGetError(), c + ": GL error");
                }
            } finally { main.destroyBuffers(); voxy.destroyBuffers(); glDeleteVertexArrays(vao); }
        } finally { glfwDestroyWindow(window); glfwTerminate(); }
    }

    private record Case(float real, float lod, boolean guard, boolean lodWins) { }

    private static void draw(GlProgram program, Matrix4f projection, float distance, float[] colour) {
        program.use();
        matrix(program.uniform("Projection"), projection);
        glUniform1f(program.uniform("Distance"), distance);
        glUniform4f(program.uniform("SurfaceColor"), colour[0], colour[1], colour[2], 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    private static float window(Matrix4f projection, float distance) {
        return Math.min(1F - 2F / 16777215F, (-projection.m22() + projection.m32() / distance) * .5F + .5F);
    }

    private static void matrix(int location, Matrix4f matrix) {
        glUniformMatrix4fv(location, false, matrix.get(new float[16]));
    }

    private static String load(ZipFile jar, String id) throws Exception {
        if (id.equals("sodium:include/fog.glsl")) return "float getFragDistance(int shape, vec3 p) { return length(p); }\n";
        String[] parts = id.split(":", 2);
        String source = new String(jar.getInputStream(jar.getEntry("assets/" + parts[0] + "/shaders/" + parts[1])).readAllBytes(), StandardCharsets.UTF_8);
        var matcher = Pattern.compile("#import <([^>]+)>").matcher(source);
        var result = new StringBuilder(); int last = 0;
        while (matcher.find()) { result.append(source, last, matcher.start()).append(load(jar, matcher.group(1))); last = matcher.end(); }
        return result.append(source.substring(last)).toString().replace("\r\n", "\n");
    }
}
