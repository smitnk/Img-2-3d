plugins { application }

java { toolchain { languageVersion.set(JavaLanguageVersion.of(17)) } }

dependencies {
    implementation(project(":core"))
    implementation(libs.onnxruntime.jvm)
}

application { mainClass.set("com.smitnk.img23d.cli.Main") }
