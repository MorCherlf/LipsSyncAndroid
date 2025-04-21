package com.morcherlf.LipsSync

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.icu.text.SimpleDateFormat
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.Image
import android.media.MediaRecorder
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope // For lifecycleScope
import com.google.common.util.concurrent.ListenableFuture
import com.google.mediapipe.framework.image.BitmapImageBuilder
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max // Import max function
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider

class MainActivity : AppCompatActivity(), FaceLandmarkerHelper.LandmarkerListener {

    // --- UI Elements ---
    private lateinit var previewView: PreviewView
    private lateinit var recordButton: Button
    private lateinit var jawOpenTextView: TextView
    private lateinit var mouthCloseTextView: TextView
    private lateinit var openFolderButton: Button

    // --- CameraX ---
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var cameraExecutor: ExecutorService // Dedicated executor for camera operations
    private var imageAnalyzer: ImageAnalysis? = null

    // --- Face Landmarker ---
    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper

    // --- Audio Recording ---
    private var audioRecord: AudioRecord? = null
    private var audioBufferSize = 0
    private val sampleRate = 16000 // Audio sample rate in Hz
    private var recordingJob: Job? = null // Coroutine job for the recording process
    private val isRecording = AtomicBoolean(false) // Thread-safe flag for recording state
    private var recordingStartTime: Long = 0L // System uptime when recording started
    private var timestampDirectory: File? = null // Directory to save recording files (audio + csv)
    private var audioFile: File? = null // Reference to the WAV file being written

    // --- Permissions Handling ---
    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            // Check if all requested permissions were granted
            val allGranted = permissions.entries.all { it.value }
            if (allGranted) {
                Log.d(TAG, "All required permissions granted.")
                // If permissions were granted, start the camera if not already started
                startCamera() // Re-check or start camera
                // If the user intended to start recording right after granting permissions
                if (shouldStartRecordingAfterPermission) {
                    startRecordingAndProcessing()
                    shouldStartRecordingAfterPermission = false // Reset the flag
                }
            } else {
                Toast.makeText(this, "Required permissions were denied.", Toast.LENGTH_LONG).show()
                shouldStartRecordingAfterPermission = false // Reset the flag as permissions were denied
            }
        }
    // Flag to indicate if recording should start immediately after permissions are granted
    private var shouldStartRecordingAfterPermission = false

    // --- Activity Lifecycle ---
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // --- Initialize UI Elements ---
        previewView = findViewById(R.id.previewView)
        recordButton = findViewById(R.id.recordButton)
        jawOpenTextView = findViewById(R.id.jawOpenTextView)
        mouthCloseTextView = findViewById(R.id.mouthCloseTextView)
        openFolderButton = findViewById(R.id.openFolderButton)

        // Initialize camera executor and provider future
        cameraExecutor = Executors.newSingleThreadExecutor()
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        // --- Initialize FaceLandmarker ---
        try {
            faceLandmarkerHelper = FaceLandmarkerHelper(
                context = this,
                faceLandmarkerHelperListener = this // MainActivity implements the listener
            )
            Log.d(TAG, "FaceLandmarkerHelper initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize FaceLandmarkerHelper: ${e.message}", e)
            Toast.makeText(this, "Failed to initialize FaceLandmarkerHelper: ${e.message}", Toast.LENGTH_LONG).show()
            // Consider disabling functionality or closing the app if initialization fails
            return
        }

        // --- Setup Record Button ---
        recordButton.setOnClickListener {
            toggleRecording() // Call the function to handle start/stop logic
        }

        openFolderButton.setOnClickListener {
            openRecordingFolder()
        }

        // --- Initial Permission Check & Camera Start ---
        // Check for camera permission to start the preview initially
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            // Request only camera permission initially
            requestPermissionsLauncher.launch(arrayOf(Manifest.permission.CAMERA))
        }

        // Initialize TextViews with default values
        updateOpennessUI(0f, 0f)
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy called.")
        // --- Release Resources ---
        // Ensure recording is stopped and resources are released
        if (isRecording.get()) {
            stopAudioRecording() // This handles stopping and releasing AudioRecord
        } else {
            // Release AudioRecord if it exists but wasn't recording (e.g., init failed)
            audioRecord?.release()
            audioRecord = null
        }
        // Shutdown camera executor
        cameraExecutor.shutdown()
        // Close FaceLandmarkerHelper
        faceLandmarkerHelper.clearFaceLandmarker()
        Log.d(TAG, "Resources released in onDestroy.")
    }


    // --- Permission Logic ---
    /**
     * Checks if the required permissions are granted. If not, requests them.
     * @param permissions Array of permissions to check (e.g., Manifest.permission.RECORD_AUDIO)
     * @return True if all permissions are already granted, false otherwise (request launched).
     */
    private fun checkAndRequestPermissions(permissions: Array<String>): Boolean {
        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }.toTypedArray()

        return if (permissionsToRequest.isEmpty()) {
            true // All permissions already granted
        } else {
            Log.d(TAG, "Requesting permissions: ${permissionsToRequest.joinToString()}")
            // Launch the permission request using the ActivityResultLauncher
            requestPermissionsLauncher.launch(permissionsToRequest)
            false // Permissions need to be granted
        }
    }

    // --- Recording Control ---
    /**
     * Toggles the recording state (Start/Stop).
     * Checks for necessary permissions before starting.
     */
    private fun toggleRecording() {
        if (isRecording.get()) {
            // If currently recording, stop it
            stopRecordingAndProcessing()
        } else {
            // If not recording, attempt to start
            // Define permissions needed specifically for recording
            val requiredPermissions = arrayOf(
                Manifest.permission.RECORD_AUDIO,
                // WRITE_EXTERNAL_STORAGE might be needed for older Android versions or specific directory access,
                // but getExternalFilesDir usually doesn't require explicit write permission.
                // Keep it for broader compatibility for now.
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            // Check if recording permissions are granted
            if (checkAndRequestPermissions(requiredPermissions)) {
                // Permissions are already granted, proceed to start recording
                startRecordingAndProcessing()
            } else {
                // Permissions are not granted, set flag to start after grant
                shouldStartRecordingAfterPermission = true
                // Permission request is launched by checkAndRequestPermissions
                Toast.makeText(this, "Requesting necessary permissions...", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /** Starts the recording process (Audio). */
    private fun startRecordingAndProcessing() {
        if (isRecording.get()) {
            Log.w(TAG, "Recording is already in progress.")
            return
        }
        Log.d(TAG, "Attempting to start recording...")
        startAudioRecording() // This function will handle the actual start and state updates
    }

    /** Stops the recording process (Audio). */
    private fun stopRecordingAndProcessing() {
        if (!isRecording.get()) {
            Log.w(TAG, "Recording is not in progress.")
            return
        }
        Log.d(TAG, "Attempting to stop recording...")
        stopAudioRecording() // This function handles the actual stop and state updates
    }


    // --- CameraX Setup and Image Processing ---
    /** Initializes and starts the camera preview and image analysis stream. */
    private fun startCamera() {
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                // Select the front camera
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                    .build()

                // --- Preview Use Case ---
                val preview = Preview.Builder()
                    // Set target aspect ratio based on view dimensions if needed,
                    // but `app:scaleType="fitCenter"` in XML usually handles visual stretching.
                    .build()
                    .apply {
                        // Link the Preview use case to the PreviewView in the layout
                        surfaceProvider = previewView.surfaceProvider
                    }

                // --- ImageAnalysis Use Case ---
                imageAnalyzer = ImageAnalysis.Builder()
                    // Analyze only the latest frame, dropping older ones if analysis is slow
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    // Request RGBA output format, required by MediaPipe
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .apply {
                        // Set the analyzer, running on the dedicated cameraExecutor background thread
                        setAnalyzer(cameraExecutor) { imageProxy ->
                            processRgbaImage(imageProxy) // Process each frame
                        }
                    }

                // Unbind any existing use cases before rebinding
                cameraProvider.unbindAll()
                // Bind the desired use cases (Preview, ImageAnalysis) to the activity's lifecycle
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                Log.d(TAG, "Camera started and bound to lifecycle.")

            } catch (e: Exception) { // Catch potential exceptions during setup
                Log.e(TAG, "Camera initialization failed", e)
                Toast.makeText(this, "Camera failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this)) // GetMainExecutor ensures listener runs on main thread
    }

    /** Processes a single frame received from the ImageAnalysis use case. */
    @OptIn(ExperimentalGetImage::class) // Needed for imageProxy.image access
    private fun processRgbaImage(imageProxy: ImageProxy) {
        try {
            // Convert the ImageProxy (YUV or other format) to a Bitmap (RGBA)
            val bitmap = imageProxy.toBitmap() ?: run {
                Log.w(TAG, "Failed to convert ImageProxy to Bitmap.")
                imageProxy.close() // Close the proxy if conversion fails
                return
            }

            // Create a MediaPipe Image object from the Bitmap
            val mpImage = BitmapImageBuilder(bitmap).build()
            // Get the timestamp associated with the frame
            val frameTime = SystemClock.uptimeMillis() // Or use imageProxy.imageInfo.timestamp for capture time

            // Perform face landmark detection asynchronously
            faceLandmarkerHelper.detectAsync(mpImage, frameTime)

            // Recycle the bitmap *after* MediaPipe is done with it.
            // IMPORTANT: If BitmapImageBuilder doesn't make a copy, recycling here is wrong.
            // Assuming for now that MediaPipe processes it quickly or copies it.
            bitmap.recycle()

        } catch (e: Exception) { // Catch errors during processing
            Log.e(TAG, "Error processing image: ${e.message}", e)
        } finally {
            // **Crucial:** Always close the ImageProxy to release the underlying image buffer
            imageProxy.close()
        }
    }

    /**
     * Extension function to convert an Image (from ImageProxy) to a Bitmap.
     * Assumes the underlying buffer contains RGBA_8888 compatible data
     * based on ImageAnalysis configuration, and handles potential row padding.
     *
     * 扩展函数：将 Image (来自 ImageProxy) 转换为 Bitmap。
     * 基于 ImageAnalysis 的配置，假定底层缓冲区包含与 RGBA_8888 兼容的数据，
     * 并处理可能的行填充。
     */
    private fun Image.toBitmap(): Bitmap? {
        // --- 移除了格式检查 ---
        // Removed format check - assuming compatible buffer based on ImageAnalysis output format request.
        // if (format != android.graphics.ImageFormat.RGBA_8888 && format != android.graphics.ImageFormat.FLEX_RGBA_8888) {
        //     Log.e(TAG,"Unsupported image format: $format, expected RGBA_8888 or FLEX_RGBA_8888")
        //     return null
        // }

        val planes = this.planes
        // Ensure there's at least one plane
        if (planes.isEmpty()) {
            Log.e(TAG, "Image has no planes.")
            return null
        }

        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride // Bytes per pixel
        val rowStride = planes[0].rowStride     // Bytes per row
        val rowPadding = rowStride - pixelStride * width // Extra bytes at the end of each row

        // Check if buffer is accessible
        if (buffer == null || pixelStride == 0 || rowStride == 0) {
            Log.e(TAG, "Image buffer or stride info is invalid.")
            return null
        }
        // Check if buffer has enough capacity
        val expectedCapacity = height * rowStride // Minimum expected capacity
        if (buffer.capacity() < expectedCapacity) {
            Log.e(TAG, "Image buffer capacity (${buffer.capacity()}) is less than expected (${expectedCapacity}).")
            // You might still try to proceed, but it indicates a potential issue.
            // Depending on the exact layout, direct copy might still work partially.
            // return null // Optionally return null if capacity seems wrong
        }


        try {
            // Create a bitmap with the raw buffer dimensions, including padding
            // Important: Use ARGB_8888 for createBitmap, as copyPixelsFromBuffer expects it for RGBA sources
            val bitmap = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888)
            buffer.rewind() // Ensure buffer position is at the start before copying
            bitmap.copyPixelsFromBuffer(buffer) // Copy pixel data

            // If there was row padding, create a correctly sized bitmap by cropping
            return if (rowPadding > 0) {
                val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height)
                bitmap.recycle() // Recycle the intermediate padded bitmap
                croppedBitmap
            } else {
                // No padding, return the bitmap directly
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during Bitmap creation or pixel copy: ${e.message}", e)
            return null // Return null if any error occurs during bitmap handling
        }
    }

    // --- FaceLandmarkerHelper Listener Callbacks ---

    override fun onError(error: String, errorCode: Int) {
        // Called when the FaceLandmarkerHelper encounters an error
        Log.e(TAG, "FaceLandmarker Error: $error (Code: $errorCode)")
        // Show error to the user on the main thread
        runOnUiThread {
            Toast.makeText(this, "Face detection error: $error", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResults(resultBundle: FaceLandmarkerHelper.ResultBundle) {
        // Called with the detection results from FaceLandmarkerHelper
        var relativeTimestampMs: Long = -1L // Use -1 to indicate invalid/not recording

        // Calculate timestamp relative to recording start time, only if recording
        if (isRecording.get() && recordingStartTime > 0L) {
            // Use the inference time provided by the helper relative to recording start
            // Ensure timestamp is not negative
            relativeTimestampMs = max(0L, resultBundle.inferenceTime - recordingStartTime)
        }

        // Process the face blendshapes if available
        resultBundle.result.faceBlendshapes().ifPresent { faceBlendshapesList ->
            // Get blendshapes for the first detected face
            val firstFaceCategories = faceBlendshapesList.firstOrNull() ?: return@ifPresent

            // Find the specific blendshape scores for jawOpen and mouthClose
            val jawOpenScore = firstFaceCategories.find { it.categoryName() == BLEND_SHAPE_JAW_OPEN }?.score() ?: 0f
            val mouthCloseScore = firstFaceCategories.find { it.categoryName() == BLEND_SHAPE_MOUTH_CLOSE }?.score() ?: 0f

            // Update the UI TextViews on the main thread
            runOnUiThread {
                updateOpennessUI(jawOpenScore, mouthCloseScore)
            }

            // Save the blendshape data to CSV only if currently recording and timestamp is valid
            if (isRecording.get() && relativeTimestampMs != -1L && timestampDirectory != null) {
                saveToCsv(relativeTimestampMs, jawOpenScore, mouthCloseScore)
            }
        }
    }

    /** Updates the TextViews displaying mouth openness scores. Must be called on the Main thread. */
    private fun updateOpennessUI(jawOpen: Float, mouthClose: Float) {
        jawOpenTextView.text = "Jaw Open: %.4f".format(jawOpen) // Format to 4 decimal places
        mouthCloseTextView.text = "Mouth Close: %.4f".format(mouthClose)
    }

    override fun onEmpty() {
        // Called when no face is detected in the frame
        Log.d(TAG, "No face detected.")
        // Optionally reset the UI display when no face is present
        runOnUiThread {
            updateOpennessUI(0f, 0f) // Reset scores to 0
        }
    }


    // --- CSV Data Saving ---
    /**
     * Saves the timestamp and blendshape scores to a CSV file.
     * Runs on a background IO thread.
     */
    private fun saveToCsv(timestampMs: Long, jawOpen: Float, mouthClose: Float) {
        val dir = timestampDirectory ?: return // Exit if directory is not set
        // Format the data as a CSV line
        val csvLine = "%d,%.4f,%.4f\n".format(timestampMs, jawOpen, mouthClose)

        // Launch a coroutine in the IO dispatcher for file operations
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val file = File(dir, "mouth_data.csv")
                // Check if file needs to be created and header written
                if (!file.exists()) {
                    try {
                        file.createNewFile()
                        // Write CSV header
                        file.appendText("timestamp_ms,jaw_open,mouth_close\n")
                        Log.d(TAG, "Created CSV file with header: ${file.absolutePath}")
                    } catch (e: IOException) {
                        Log.e(TAG, "Failed to create or write header for CSV: ${file.absolutePath}", e)
                        return@launch // Stop if creating/writing header fails
                    }
                }
                // Append the data line to the file
                file.appendText(csvLine)
            } catch (e: Exception) { // Catch potential errors during appendText
                Log.e(TAG, "Failed to save data to CSV: ${e.message}", e)
            }
        }
    }


    // --- Audio Recording Logic ---
    /** Initializes and starts the audio recording process. */
    @SuppressLint("MissingPermission") // Permissions are checked before calling this function
    private fun startAudioRecording() {
        // 1. Calculate minimum buffer size
        audioBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        // Check if buffer size calculation was successful
        if (audioBufferSize == AudioRecord.ERROR || audioBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "AudioRecord: Failed to get minimum buffer size.")
            Toast.makeText(this, "AudioRecord Error: Invalid buffer size.", Toast.LENGTH_SHORT).show()
            return // Stop if buffer size is invalid
        }

        // 2. Create timestamped directory for saving files
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        timestampDirectory = File(getExternalFilesDir(null), timestamp).apply {
            // Try to create the directory if it doesn't exist
            if (!exists() && !mkdirs()) {
                Log.e(TAG, "Failed to create directory: $absolutePath")
                Toast.makeText(applicationContext, "Error: Cannot create directory.", Toast.LENGTH_SHORT).show()
                timestampDirectory = null // Reset if creation failed
                return // Stop if directory cannot be created
            }
        }
        Log.d(TAG, "Created recording directory: ${timestampDirectory?.absolutePath}")

        // 3. Initialize AudioRecord instance
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC, // Use microphone as audio source
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            audioBufferSize * 2 // Use a buffer slightly larger than the minimum
        ).apply {
            // Check if initialization was successful
            if (state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed. State: $state")
                Toast.makeText(applicationContext, "AudioRecord Error: Initialization failed.", Toast.LENGTH_SHORT).show()
                audioRecord = null // Ensure it's null if failed
                return // Stop if initialization fails
            }
        }

        // 4. Prepare the output WAV file path
        audioFile = File(timestampDirectory, "audio.wav")

        // 5. Record the start time (using monotonic clock)
        recordingStartTime = SystemClock.uptimeMillis()

        // 6. Start the recording process in a background coroutine
        recordingJob = lifecycleScope.launch(Dispatchers.IO) { // Use IO dispatcher for file operations
            var fileOutputStream: FileOutputStream? = null // Use local variable for stream
            var totalBytesWritten: Long = 0 // Track total bytes for WAV header

            try {
                // Open the output file stream
                fileOutputStream = FileOutputStream(audioFile!!) // Safe non-null due to previous checks
                // Write a placeholder WAV header (44 bytes of zeros) initially
                writeWavHeaderPlaceholder(fileOutputStream)

                // Start the actual audio capture
                audioRecord?.startRecording()
                // Verify recording started successfully
                if (audioRecord?.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                    throw IOException("AudioRecord failed to start recording.") // Throw exception if failed
                }

                // Update recording state and UI on the Main thread *after* successful start
                withContext(Dispatchers.Main) {
                    isRecording.set(true)
                    recordButton.text = "Stop Recording" // Update button text
                    Log.d(TAG, "Audio recording started successfully.")
                }

                // --- Recording Loop ---
                val audioBuffer = ByteArray(audioBufferSize) // Reuse buffer
                // Continue loop while coroutine is active and recording flag is set
                while (isActive && isRecording.get()) {
                    // Read audio data from AudioRecord buffer
                    val bytesRead = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: -1
                    if (bytesRead > 0) {
                        // Write read data to the file output stream
                        try {
                            fileOutputStream.write(audioBuffer, 0, bytesRead)
                            totalBytesWritten += bytesRead // Update total bytes count
                        } catch (e: IOException) {
                            Log.e(TAG, "Error writing audio chunk to file", e)
                            break // Exit loop on write error
                        }
                    } else if (bytesRead < 0) {
                        // Handle read errors
                        Log.e(TAG, "AudioRecord read error: $bytesRead")
                        break // Exit loop on read error
                    }
                    // yield() // Optional: allow other coroutines to run
                } // --- End of Recording Loop ---

                Log.d(TAG, "Audio recording loop finished. Total bytes written: $totalBytesWritten")

                // --- Finalize WAV Header ---
                // Update the placeholder WAV header with correct file sizes
                try {
                    // Check if file exists before updating header
                    if (audioFile?.exists() == true) {
                        updateWavHeader(audioFile!!, totalBytesWritten)
                        Log.d(TAG, "WAV header updated successfully for ${audioFile?.name}")
                    } else {
                        Log.w(TAG, "Audio file does not exist, cannot update WAV header.")
                    }

                } catch (e: IOException) {
                    Log.e(TAG, "Failed to update WAV header", e)
                }

            } catch (e: Exception) { // Catch errors during setup or recording loop
                Log.e(TAG, "Audio recording failed: ${e.message}", e)
                // Ensure UI/state is reset correctly on the Main thread if an error occurs
                withContext(Dispatchers.Main) {
                    if (isRecording.get()) { // Check if it was thought to be recording
                        isRecording.set(false) // Reset state flag
                        recordButton.text = "Start Recording" // Reset button text
                        Toast.makeText(applicationContext, "Recording error: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                    }
                }
            } finally {
                // --- Cleanup within Coroutine ---
                Log.d(TAG, "Audio recording coroutine finally block.")
                // 1. Stop AudioRecord if it's currently recording
                if (audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    try {
                        audioRecord?.stop()
                        Log.d(TAG, "AudioRecord stopped in finally.")
                    } catch (e: IllegalStateException) {
                        // Can happen if stop() is called in an invalid state
                        Log.e(TAG, "Error stopping AudioRecord in finally", e)
                    }
                }
                // Note: AudioRecord.release() is called in stopAudioRecording on the main thread

                // 2. Close the FileOutputStream
                try {
                    fileOutputStream?.close()
                    Log.d(TAG, "FileOutputStream closed in finally.")
                } catch (e: IOException) {
                    Log.e(TAG, "Error closing FileOutputStream in finally", e)
                }
            }
        } // --- End of Recording Coroutine ---
    } // --- End of startAudioRecording ---

    /** Stops the audio recording process and releases resources. */
    private fun stopAudioRecording() {
        if (!isRecording.get()) return // Do nothing if not recording

        isRecording.set(false) // Signal the recording loop (running in coroutine) to stop

        // Cancel the coroutine job handling the recording loop and file writing
        recordingJob?.cancel() // Request cancellation
        recordingJob = null    // Clear the job reference

        // Use lifecycleScope to ensure release happens even if activity is finishing
        lifecycleScope.launch {
            // Optional: Wait for the job to fully complete (including finally block) if needed
            // recordingJob?.join()

            // Release the AudioRecord instance
            try {
                // Check state before releasing, though release() should be safe
                audioRecord?.release()
                Log.d(TAG, "AudioRecord released.")
            } catch (e: Exception) { // Catch potential errors during release
                Log.e(TAG, "Error releasing AudioRecord", e)
            }
            audioRecord = null // Clear the reference

            // Update UI and reset state variables on the Main thread
            withContext(Dispatchers.Main) {
                recordButton.text = "Start Recording" // Reset button text
                recordingStartTime = 0L // Reset start time
                // Optionally clear directory/file references if needed
                // timestampDirectory = null
                // audioFile = null
                Log.d(TAG, "Audio recording stopped and resources released.")
            }
        }
    }


    // --- WAV File Header Handling ---
    /** Writes a 44-byte placeholder (zeros) to the beginning of the WAV file. */
    @Throws(IOException::class)
    private fun writeWavHeaderPlaceholder(stream: FileOutputStream) {
        stream.write(ByteArray(44)) // Write 44 zero bytes
    }

    /**
     * Updates the WAV file header with the correct chunk sizes after recording is complete.
     * @param file The WAV file to update.
     * @param totalAudioLen The total number of audio data bytes written (excluding header).
     */
    @Throws(IOException::class)
    private fun updateWavHeader(file: File, totalAudioLen: Long) {
        // Calculate file size fields for the header
        val totalDataLen = totalAudioLen + 36 // RIFF chunk size = total file size - 8 bytes ("RIFF", "WAVE")
        val channels = 1 // Mono
        val bitDepth: Short = 16 // 16-bit PCM
        val byteRate = sampleRate * channels * (bitDepth / 8) // Bytes per second

        // Use RandomAccessFile to overwrite the header at the beginning of the file
        RandomAccessFile(file, "rw").use { raf -> // "rw" mode for read/write access
            // Prepare the 44-byte header in Little Endian format
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

            // RIFF Header
            header.put("RIFF".toByteArray())           // ChunkID (4 bytes)
            header.putInt(totalDataLen.toInt())        // ChunkSize (4 bytes) - File size minus first 8 bytes
            header.put("WAVE".toByteArray())           // Format (4 bytes)

            // Format Subchunk ("fmt ")
            header.put("fmt ".toByteArray())           // Subchunk1ID (4 bytes)
            header.putInt(16)                          // Subchunk1Size (4 bytes) - 16 for PCM
            header.putShort(1)                         // AudioFormat (2 bytes) - 1 for PCM
            header.putShort(channels.toShort())        // NumChannels (2 bytes)
            header.putInt(sampleRate)                  // SampleRate (4 bytes)
            header.putInt(byteRate)                    // ByteRate (4 bytes) - SampleRate * NumChannels * BitsPerSample/8
            header.putShort((channels * (bitDepth / 8)).toShort()) // BlockAlign (2 bytes) - NumChannels * BitsPerSample/8
            header.putShort(bitDepth)                  // BitsPerSample (2 bytes)

            // Data Subchunk ("data")
            header.put("data".toByteArray())           // Subchunk2ID (4 bytes)
            header.putInt(totalAudioLen.toInt())       // Subchunk2Size (4 bytes) - Size of actual audio data

            // Go to the beginning of the file
            raf.seek(0)
            // Write the generated header bytes
            raf.write(header.array())
            Log.d(TAG, "WAV header updated: File=${file.name}, AudioLen=$totalAudioLen, TotalLen=$totalDataLen")
        } // RandomAccessFile is automatically closed by .use {}
    }


    // TODO
    // 打开录制文件夹
    // 可用性待确认：无法直接授予Action打开的Activity当前应用的文件权限（？）
    private fun openRecordingFolder() {
        // 1. 获取应用的外部文件存储根目录 (包含所有时间戳子文件夹)
        val rootDir = getExternalFilesDir(null)

        // 2. 检查此目录是否存在 (通常总是存在，除非外部存储不可用)
        if (rootDir == null || !rootDir.exists()) {
            Toast.makeText(this, "Application Data Folder Not Found", Toast.LENGTH_SHORT).show()
            Log.w(TAG, "getExternalFilesDir(null) returned null or does not exist.")
            return
        }

        // 3. 使用 FileProvider 为这个根目录创建 content URI
        // FileProvider 的 authority 需要和 Manifest 中定义的一致
        val authority = "${applicationContext.packageName}.provider"
        val contentUri: Uri? = try {
            // 为根目录获取 URI
            FileProvider.getUriForFile(this, authority, rootDir)
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "FileProvider setup error for $rootDir. Check authorities and file_paths.xml.", e)
            Toast.makeText(this, "Unreachable Folder URI", Toast.LENGTH_LONG).show()
            null
        }

        // 如果获取 URI 失败，则返回
        if (contentUri == null) {
            return
        }

        // 4. 创建 ACTION_VIEW Intent 来查看这个目录
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            // 设置根目录的 URI
            // 同样，文件夹的 MIME 类型不标准，尝试通用类型或只设置数据
            setDataAndType(contentUri, "*/*") // 尝试通用类型


            // 授予读取权限给将要打开此 URI 的应用
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

        // 5. 检查是否有应用可以处理此 Intent，然后启动它
        val packageManager = packageManager
        if (intent.resolveActivity(packageManager) != null) {
            try {
                startActivity(intent)
                Log.d(TAG, "Launched ACTION_VIEW intent for root data folder: $contentUri")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start activity for ACTION_VIEW with root folder URI", e)
                Toast.makeText(this, "Failed to open folder, something wrong when open activity", Toast.LENGTH_SHORT).show()
            }
        } else {
            // 没有找到可以处理此 Intent 的应用 (例如，没有安装文件管理器)
            Log.w(TAG, "No activity found to handle ACTION_VIEW for folder: $contentUri")
            Toast.makeText(this, "No application could open the folder", Toast.LENGTH_SHORT).show()
        }
    }

    // --- Companion Object ---
    companion object {
        // Log Tag for identifying logs from this activity
        private const val TAG = "MainActivityLipsSync"
        // Constant names for blendshapes being tracked
        private const val BLEND_SHAPE_JAW_OPEN = "jawOpen"
        private const val BLEND_SHAPE_MOUTH_CLOSE = "mouthClose"
        // Permission request codes are handled by ActivityResultContracts, no need for separate constants here
    }
}