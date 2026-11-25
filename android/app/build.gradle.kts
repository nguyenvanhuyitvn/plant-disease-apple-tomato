plugins {
    id("com.android.application")
}

android {
    namespace = "vn.plantdx"
    compileSdk = 35

    defaultConfig {
        applicationId = "vn.plantdx.species"   // khác ứng dụng 27 lớp cũ, cài song song được
        minSdk = 28            // Android 9 trở lên (ImageDecoder tự xoay ảnh theo EXIF)
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }   // điện thoại thật
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Ký bằng khoá debug để cài trực tiếp (sideload); khi đưa lên Play Store cần khoá riêng.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources { noCompress += "onnx" }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")
    implementation("androidx.core:core:1.13.1")   // FileProvider cho ảnh chụp từ camera
}
