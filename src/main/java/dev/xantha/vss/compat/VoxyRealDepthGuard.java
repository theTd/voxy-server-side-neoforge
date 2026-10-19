package dev.xantha.vss.compat;

import dev.xantha.vss.common.VSSLogger;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL12;
import org.lwjgl.opengl.GL13;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL33;
import org.lwjgl.opengl.GL43;
import org.lwjgl.opengl.GL45;

/**
 * Keeps Voxy's final blit from replacing the depth of real geometry it is not clearly in front of.
 *
 * <p>VSS enables depth writes for normal Voxy's final blit so later passes are occluded by LOD
 * terrain. Voxy's vanilla coverage mask, however, follows world sections only: sub-level mods
 * (Sable structures) render real blocks over sections Voxy treats as uncovered, and a stale LOD
 * copy coplanar with those blocks then wrote its coarser depth over them. Decals drawn a few
 * millimetres in front of such faces (drawer item/count/lock overlays) failed the depth test
 * except at very close range.
 *
 * <p>The blit now compares each LOD fragment with a copy of the real depth taken just before the
 * blit. Over sky the LOD keeps writing depth as before; over real geometry it is only drawn when it
 * lies in front by a distance-scaled margin, so LOD hills still hide distant real geometry while
 * coplanar or stale LOD can no longer cover it.
 */
public final class VoxyRealDepthGuard {
    private static final String PATH = "voxy:post/blit_texture_depth_cutout.frag";
    private static final String MARKER = "VssRealDepthGuard";
    private static final String WRITE = "gl_FragDepth = depth;";
    private static final int UNIT = 6;
    private static final boolean DISABLED = Boolean.getBoolean("vss.disableVoxyRealDepthGuard");

    private static int copyTexture;
    private static int copyFormat;
    private static int copyWidth;
    private static int copyHeight;
    private static boolean failed;
    private static boolean frameReady;
    private static boolean bound;
    private static int savedTexture;
    private static int savedSampler;

    private VoxyRealDepthGuard() {
    }

    public static String patch(String path, String source) {
        if (DISABLED || !PATH.equals(path) || source == null || source.contains(MARKER)) return source;
        // Only the known final-blit contract is changed; unknown layouts keep Voxy's behaviour.
        if (!source.contains(WRITE) || !source.contains("uniform mat4 projMat")
                || !source.contains("vec3 point = rev3d(") || !source.contains("void main()")) return source;
        return source.replace("void main()", """
                uniform bool VssRealDepthGuard;
                uniform sampler2D VssRealDepth;
                void main()""")
                .replace(WRITE, """
                if (VssRealDepthGuard) {
                    float realDepth = texelFetch(VssRealDepth, ivec2(gl_FragCoord.xy), 0).r;
                    if (realDepth < 1.0) {
                        // Real geometry wins unless the LOD is clearly closer: push the LOD
                        // fragment away by a margin that grows with LOD cell size and test again.
                        float lodDistance = max(length(point), 1.0e-4);
                        float margin = max(1.0, lodDistance * (1.0 / 64.0));
                        vec4 pushed = projMat * vec4(point * (1.0 + margin / lodDistance), 1.0);
                        float pushedDepth = pushed.w > 0.0 ? (pushed.z / pushed.w) * 0.5 + 0.5 : 1.0;
                        if (pushedDepth >= realDepth) discard;
                    }
                }
                """ + WRITE);
    }

    /** Copies the destination depth before Voxy's final blit starts writing to it. */
    public static void prepare(int targetFramebuffer) {
        frameReady = false;
        if (DISABLED || failed || targetFramebuffer <= 0) return;
        try {
            int type = GL45.glGetNamedFramebufferAttachmentParameteri(targetFramebuffer, GL30.GL_DEPTH_ATTACHMENT,
                    GL30.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE);
            if (type != GL11.GL_TEXTURE) return;
            int source = GL45.glGetNamedFramebufferAttachmentParameteri(targetFramebuffer, GL30.GL_DEPTH_ATTACHMENT,
                    GL30.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
            if (source <= 0) return;
            int format = GL45.glGetTextureLevelParameteri(source, 0, GL11.GL_TEXTURE_INTERNAL_FORMAT);
            int width = GL45.glGetTextureLevelParameteri(source, 0, GL11.GL_TEXTURE_WIDTH);
            int height = GL45.glGetTextureLevelParameteri(source, 0, GL11.GL_TEXTURE_HEIGHT);
            if (width <= 0 || height <= 0) return;
            if (copyTexture == 0 || format != copyFormat || width != copyWidth || height != copyHeight) {
                if (copyTexture != 0) GL11.glDeleteTextures(copyTexture);
                copyTexture = GL45.glCreateTextures(GL11.GL_TEXTURE_2D);
                allocateLike(copyTexture, format, width, height);
                GL45.glTextureParameteri(copyTexture, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_NEAREST);
                GL45.glTextureParameteri(copyTexture, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_NEAREST);
                GL45.glTextureParameteri(copyTexture, GL30.GL_TEXTURE_COMPARE_MODE, GL11.GL_NONE);
                if (format == GL30.GL_DEPTH24_STENCIL8 || format == GL30.GL_DEPTH32F_STENCIL8)
                    GL45.glTextureParameteri(copyTexture, GL43.GL_DEPTH_STENCIL_TEXTURE_MODE, GL11.GL_DEPTH_COMPONENT);
                copyFormat = format;
                copyWidth = width;
                copyHeight = height;
            }
            GL43.glCopyImageSubData(source, GL11.GL_TEXTURE_2D, 0, 0, 0, 0,
                    copyTexture, GL11.GL_TEXTURE_2D, 0, 0, 0, 0, width, height, 1);
            frameReady = true;
        } catch (RuntimeException failure) {
            fail(failure);
        }
    }

    /** Called while Voxy's final-blit program is bound, right before the blit. */
    public static void bind() {
        int program = GL11.glGetInteger(GL20.GL_CURRENT_PROGRAM);
        int enabled = GL20.glGetUniformLocation(program, MARKER);
        if (enabled < 0) return;
        boolean active = frameReady && !failed;
        GL20.glUniform1i(enabled, active ? 1 : 0);
        if (!active) return;
        int active0 = GL11.glGetInteger(GL13.GL_ACTIVE_TEXTURE);
        GL13.glActiveTexture(GL13.GL_TEXTURE0 + UNIT);
        savedTexture = GL11.glGetInteger(GL11.GL_TEXTURE_BINDING_2D);
        GL13.glActiveTexture(active0);
        savedSampler = GL30.glGetIntegeri(GL33.GL_SAMPLER_BINDING, UNIT);
        GL45.glBindTextureUnit(UNIT, copyTexture);
        GL33.glBindSampler(UNIT, 0);
        GL20.glUniform1i(GL20.glGetUniformLocation(program, "VssRealDepth"), UNIT);
        bound = true;
    }

    /** Restores the texture unit borrowed by {@link #bind()}. */
    public static void restore() {
        frameReady = false;
        if (!bound) return;
        bound = false;
        GL45.glBindTextureUnit(UNIT, savedTexture);
        GL33.glBindSampler(UNIT, savedSampler);
    }

    /**
     * glCopyImageSubData requires matching depth formats. Minecraft allocates its depth with the
     * unsized GL_DEPTH_COMPONENT, which immutable storage rejects, so unsized formats are mirrored
     * with the same mutable allocation call.
     */
    private static void allocateLike(int texture, int format, int width, int height) {
        switch (format) {
            case GL11.GL_DEPTH_COMPONENT, GL30.GL_DEPTH_STENCIL -> {
                boolean stencil = format == GL30.GL_DEPTH_STENCIL;
                int previous = GL11.glGetInteger(GL11.GL_TEXTURE_BINDING_2D);
                GL11.glBindTexture(GL11.GL_TEXTURE_2D, texture);
                GL11.glTexImage2D(GL11.GL_TEXTURE_2D, 0, format, width, height, 0, format,
                        stencil ? GL30.GL_UNSIGNED_INT_24_8 : GL11.GL_FLOAT, (java.nio.ByteBuffer) null);
                GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL12.GL_TEXTURE_MAX_LEVEL, 0);
                GL11.glBindTexture(GL11.GL_TEXTURE_2D, previous);
            }
            default -> GL45.glTextureStorage2D(texture, 1, format, width, height);
        }
    }

    private static void fail(RuntimeException failure) {
        failed = true;
        frameReady = false;
        VSSLogger.warn("Voxy real-depth guard unavailable; LOD depth may cover sub-level geometry", failure);
    }
}
