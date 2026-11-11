plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.smitnk.img23d"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.smitnk.img23d"
        minSdk = 26
        // Blender (System A) runs as a Linux binary under proot from app storage; Android 10+ only allows exec of
        // app-data files for apps targeting <= 28. Sideload-only by design.
        @Suppress("ExpiredTargetSdkVersion")
        targetSdk = 28
        versionCode = 2
        versionName = "2.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a") }
        buildConfigField("String", "RUNTIME_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/blender-runtime-v1/\"")
        buildConfigField("String", "BODY_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/body-assets-v1/\"")
        buildConfigField("String", "TEXGEN_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/texgen-v1/\"")
        buildConfigField("String", "HUNYUAN_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/hunyuan-v1/\"")
        buildConfigField("String", "CURIO_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/curiomesh-v1/\"")
        buildConfigField("String", "HUNYUANMV_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/hunyuanmv-v1/\"")
        buildConfigField("String", "MODEL_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/models-v1/\"")
    }

    signingConfigs {
        // One fixed key for every CI build, so a new APK installs over the old one and keeps the downloaded engine packs
        // (a fresh runner-generated debug key forced an uninstall, which wiped them).
        // shortcut: this key is in the repo (debug-grade, standard "android" passwords); use a private keystore before publishing.
        create("ci") {
            storeFile = file("ci-debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        debug { signingConfig = signingConfigs.getByName("ci") }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("ci")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    packaging { jniLibs { useLegacyPackaging = false } }
    lint { disable += setOf("ExpiredTargetSdkVersion", "OldTargetApi") }
}

dependencies {
    implementation(project(":viewer"))
    implementation(libs.onnxruntime.android)
    implementation(libs.commons.compress)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.compose.ui.tooling.preview)
    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.junit)
    testImplementation("org.json:json:20240303") // real org.json for JVM unit tests (Android's is a stub)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext)
    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.compose.ui.test.junit4)
    debugImplementation(libs.compose.ui.test.manifest)
}
