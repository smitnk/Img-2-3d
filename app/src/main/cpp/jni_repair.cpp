#include <jni.h>

#include "mesh_repair.h"

// Object[] { float[] positions, int[] indices, float[] colors | null, int[] stats, String status, double volume }
// stats = trisBefore, trisAfter, vertsBefore, vertsAfter, watertight(0/1), piecesRemoved, holesFilled, welded, degenerate
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_smitnk_img23d_repair_MeshRepair_nativeRepair(JNIEnv* env, jclass, jfloatArray jpos, jintArray jidx, jfloatArray jcol,
                                                      jdouble weldRel, jdouble minPiecePct) {
    const jsize np = env->GetArrayLength(jpos), ni = env->GetArrayLength(jidx);
    jfloat* pos = env->GetFloatArrayElements(jpos, nullptr);
    jint* idx = env->GetIntArrayElements(jidx, nullptr);
    jfloat* col = jcol ? env->GetFloatArrayElements(jcol, nullptr) : nullptr;
    const bool colOk = col && env->GetArrayLength(jcol) == np;

    repair::Params p;
    p.weldRel = weldRel;
    p.minPiecePct = minPiecePct;
    repair::Stats st;
    repair::Mesh r = repair::run(pos, np / 3, idx, ni, colOk ? col : nullptr, p, st);

    env->ReleaseFloatArrayElements(jpos, pos, JNI_ABORT);
    env->ReleaseIntArrayElements(jidx, idx, JNI_ABORT);
    if (col) env->ReleaseFloatArrayElements(jcol, col, JNI_ABORT);

    jclass obj = env->FindClass("java/lang/Object");
    jobjectArray out = env->NewObjectArray(6, obj, nullptr);
    jfloatArray op = env->NewFloatArray((jsize)r.pos.size());
    env->SetFloatArrayRegion(op, 0, (jsize)r.pos.size(), r.pos.data());
    jintArray oi = env->NewIntArray((jsize)r.idx.size());
    env->SetIntArrayRegion(oi, 0, (jsize)r.idx.size(), r.idx.data());
    env->SetObjectArrayElement(out, 0, op);
    env->SetObjectArrayElement(out, 1, oi);
    if (!r.col.empty()) {
        jfloatArray oc = env->NewFloatArray((jsize)r.col.size());
        env->SetFloatArrayRegion(oc, 0, (jsize)r.col.size(), r.col.data());
        env->SetObjectArrayElement(out, 2, oc);
    }
    jint s[9] = {st.trisBefore, st.trisAfter, st.vertsBefore, st.vertsAfter, st.watertight ? 1 : 0,
                 st.piecesRemoved, st.holesFilled, st.welded, st.degenerate};
    jintArray os = env->NewIntArray(9);
    env->SetIntArrayRegion(os, 0, 9, s);
    env->SetObjectArrayElement(out, 3, os);
    env->SetObjectArrayElement(out, 4, env->NewStringUTF(st.status.c_str()));
    jclass dcls = env->FindClass("java/lang/Double");
    jmethodID dv = env->GetStaticMethodID(dcls, "valueOf", "(D)Ljava/lang/Double;");
    env->SetObjectArrayElement(out, 5, env->CallStaticObjectMethod(dcls, dv, (jdouble)st.volume));
    return out;
}
