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
        buildConfigField("String", "MODEL_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/models-v1/\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // shortcut: release is signed with the debug key so CI produces an installable APK; add a real keystore before publishing.
            signingConfig = signingConfigs.getByName("debug")
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
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext)
    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.compose.ui.test.junit4)
    debugImplementation(libs.compose.ui.test.manifest)
}
