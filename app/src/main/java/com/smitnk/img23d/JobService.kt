package com.smitnk.img23d

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.PowerManager

/** Keeps the process foreground-important (and the CPU awake) while a job runs, so the low-memory killer takes other apps first. */
class JobService : Service() {
    private var wake: PowerManager.WakeLock? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel("job", "Running jobs", NotificationManager.IMPORTANCE_LOW))
        val n = Notification.Builder(this, "job").setSmallIcon(applicationInfo.icon).setOngoing(true)
            .setContentTitle("Img-2-3D is working: ${intent?.getStringExtra("title") ?: ""}").build()
        startForeground(1, n) // targetSdk 28: no foregroundServiceType needed
        if (wake == null) wake = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "img23d:job").apply { acquire(4 * 60 * 60 * 1000L) }
        return START_NOT_STICKY
    }

    override fun onDestroy() { wake?.release() }

    override fun onBind(intent: Intent?): IBinder? = null
}
