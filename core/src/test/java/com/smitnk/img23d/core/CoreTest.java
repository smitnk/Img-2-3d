package com.smitnk.img23d.core;

import org.junit.Test;

public class CoreTest {
    @Test public void sphere() { CoreSelfCheck.sphereIsWatertightAndRound(); }
    @Test public void glb() { CoreSelfCheck.glbIsWellFormed(); }
    @Test public void resize() { CoreSelfCheck.resizeMatchesTorchSemantics(); }
    @Test public void foreground() { CoreSelfCheck.resizeForegroundCentersObject(); }
}
