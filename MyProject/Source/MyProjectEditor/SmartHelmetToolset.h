#pragma once

#include "CoreMinimal.h"
#include "ToolsetRegistry/ToolsetDefinition.h"
#include "ToolsetRegistry/ToolsetImage.h"
#include "SmartHelmetToolset.generated.h"

/**
 * Smart Helmet: 8-camera panoramic helmet (7 ring cameras + 1 top camera, IMX462 fisheye 141 deg, BNO055 IMU).
 * Tools capture synchronized frames from the helmet in the running PIE session (or the editor level),
 * return rig calibration, control periodic capture and stitch OpenCV panoramas via the Python pipeline.
 */
UCLASS()
class USmartHelmetToolset : public UToolsetDefinition
{
	GENERATED_BODY()

public:
	/**
	 * Capture one synchronized frame from all helmet cameras (writes cam0..cam6.png, top7.png and meta.json with the IMU record).
	 * Uses the Play-In-Editor world if PIE is running, otherwise the editor level.
	 * @return JSON: {"frame_dir", "session_dir", "frame", "world", "imu"} or {"error"}.
	 */
	UFUNCTION(meta = (AICallable))
	static FString CaptureFrame();

	/**
	 * Return the helmet rig calibration (rig.json format used by the OpenCV stitcher): intrinsics, positions and rotations of the 8 cameras.
	 * @return rig JSON or {"error"}.
	 */
	UFUNCTION(meta = (AICallable))
	static FString GetRigCalibration();

	/**
	 * Enable or disable periodic capture (test programme: one synchronized photo every 5 s during a shift).
	 * @param bEnabled true to start periodic capture.
	 * @param IntervalSeconds capture period in seconds (>= 0.2).
	 * @return JSON status.
	 */
	UFUNCTION(meta = (AICallable))
	static FString SetAutoCapture(bool bEnabled, float IntervalSeconds);

	/**
	 * Configure capture quality.
	 * @param FaceResolution cube-face resolution used to render each fisheye image (256..4096; 1024 default).
	 * @param ExposureEV100 locked exposure for all cameras (EV100, e.g. 9 indoor, 11-14 daylight).
	 * @param MountErrorStdDeg simulated camera installation error (deg, 0 = ideal helmet).
	 * @return JSON status.
	 */
	UFUNCTION(meta = (AICallable))
	static FString ConfigureCapture(int32 FaceResolution, float ExposureEV100, float MountErrorStdDeg);

	/**
	 * Simulate real-hardware imperfections per camera (new random draw with the given seed). All zeros = ideal helmet.
	 * @param Seed random seed (same seed = same faults).
	 * @param MountErrorStdDeg camera installation error, std in degrees (real helmet: 3-10 deg).
	 * @param LensKStd lens-to-lens spread of Kannala-Brandt k1 (k2..k4 get 1/2, 1/4, 1/8), e.g. 0.03.
	 * @param TriggerJitterMaxS max trigger delay of slave cameras after the master, seconds (real: up to 1 s). Needs PIE.
	 * @param ExposureStdStops independent auto-exposure spread, stops (real: ~1.0).
	 * @param WhiteBalanceStd AWB spread (std of log channel gains), e.g. 0.1.
	 * @param VignetteStrength 0..1 cos^4 lens fall-off.
	 * @param NoiseStd sensor noise in 8-bit units, e.g. 3.
	 * @param bCaptureReference also save an ideal 360 reference (ground truth) for every frame.
	 * @return JSON with the drawn per-camera truth.
	 */
	UFUNCTION(meta = (AICallable))
	static FString ConfigureImperfections(int32 Seed, float MountErrorStdDeg, float LensKStd, float TriggerJitterMaxS,
		float ExposureStdStops, float WhiteBalanceStd, float VignetteStrength, float NoiseStd, bool bCaptureReference);

	/**
	 * Let the operator walk in a circle in PIE so that unsynchronised triggers see a moving head.
	 * @param bEnabled walk on/off.
	 * @param WalkInput 0..1 forward input (0.5 ~ 1.5 m/s).
	 * @param TurnDegPerSec heading change rate, deg/s.
	 * @return JSON status.
	 */
	UFUNCTION(meta = (AICallable))
	static FString SetOperatorMotion(bool bEnabled, float WalkInput, float TurnDegPerSec);

	/**
	 * Status of the helmet: world, session folder, frame count, auto-capture state, current IMU record.
	 * @return JSON status.
	 */
	UFUNCTION(meta = (AICallable))
	static FString GetStatus();

	/**
	 * Stitch a captured frame into an equirectangular panorama with the OpenCV pipeline (Tools/smart_helmet_pano).
	 * @param FrameDir frame folder; empty = last captured frame.
	 * @param Width panorama width in pixels (height = width/2), e.g. 2048 or 4096.
	 * @return JSON with panorama path, processing time and artifact metric (pass if < 10 % of central area).
	 */
	UFUNCTION(meta = (AICallable))
	static FString StitchFrame(const FString& FrameDir, int32 Width);

	/**
	 * Load an image from disk (e.g. a stitched panorama.jpg or a camera png) to view it.
	 * @param Path absolute file path.
	 * @return The image.
	 */
	UFUNCTION(meta = (AICallable))
	static FToolsetImage ViewImage(const FString& Path);
};
