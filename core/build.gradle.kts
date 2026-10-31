plugins { `java-library` }

java { toolchain { languageVersion.set(JavaLanguageVersion.of(17)) } }

dependencies {
    // Same ai.onnxruntime API ships in onnxruntime-android; each consumer picks its runtime.
    compileOnly(libs.onnxruntime.jvm)
    testImplementation(libs.junit)
}
