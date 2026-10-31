package com.smitnk.img23d

import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HomeScreenTest {
    @get:Rule val rule = createAndroidComposeRule<MainActivity>()

    @Test fun homeOffersImagePicker() {
        rule.onNodeWithText("Photo → 3D").assertExists()
    }
}
