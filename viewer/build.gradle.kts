plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.smitnk.img23d.viewer"
    compileSdk = 37
    defaultConfig { minSdk = 26 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    api(project(":core"))
    testImplementation(libs.junit)
    testImplementation("org.json:json:20240303") // real org.json for JVM unit tests (Android's is a stub)
}
