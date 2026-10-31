plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.smitnk.forge3d"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.smitnk.forge3d"
        minSdk = 26
        // Blender runs as a Linux binary under proot from app storage; Android 10+ only allows exec of
        // app-data files for apps targeting <= 28. Sideload-only by design (see README, System A).
        @Suppress("ExpiredTargetSdkVersion")
        targetSdk = 28
        versionCode = 1
        versionName = "1.0.0"
        ndk { abiFilters += listOf("arm64-v8a") }
        buildConfigField("String", "RUNTIME_BASE_URL", "\"https://github.com/smitnk/Img-2-3d/releases/download/blender-runtime-v1/\"")
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            // shortcut: signed with the debug key so CI produces an installable APK; add a real keystore before distributing.
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
    lint { disable += setOf("ExpiredTargetSdkVersion", "OldTargetApi") }
}

dependencies {
    implementation(project(":viewer"))
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
}
