// Smart Helmet: 8-camera panoramic helmet (7 ring + 1 top) with BNO055-style IMU.
// Geometry follows "Scheme_Helmet_8_Cameras": ring yaw step 51.43 deg, axis pitch +9.3 deg,
// portrait IMX462 1080x1920, VFOV 141.4 deg equidistant fisheye; top camera at the zenith.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "SmartHelmetComponent.generated.h"

class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
class UStaticMeshComponent;

USTRUCT(BlueprintType)
struct FHelmetCameraSpec
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FString Name;

	/** Lens position relative to the helmet centre, cm (X fwd, Y right, Z up). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FVector Position = FVector::ZeroVector;

	/** Optical axis = X of this rotation. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	FRotator Rotation = FRotator::ZeroRotator;
};

/** Per-camera hardware imperfections (truth, not known to the stitcher). */
USTRUCT(BlueprintType)
struct FHelmetCameraFault
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fault")
	FRotator MountError = FRotator::ZeroRotator;

	/** Added to the component LensK for this camera (lens-to-lens spread). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fault")
	FVector4 LensKDelta = FVector4(0, 0, 0, 0);

	/** Trigger delay after the master (s): unsynchronised Raspberry Pis. Applied only in a ticking world. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fault")
	float TriggerDelay = 0.f;

	/** Independent auto-exposure: linear gain. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fault")
	float ExposureGain = 1.f;

	/** Independent AWB: per-channel linear gains. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fault")
	FLinearColor WhiteBalance = FLinearColor::White;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnHelmetFrameCaptured, int32, FrameIndex, const FString&, FrameDir);

UCLASS(ClassGroup = (SmartHelmet), meta = (BlueprintSpawnableComponent), HideCategories = (Mobility))
class MYPROJECT_API USmartHelmetComponent : public USceneComponent
{
	GENERATED_BODY()

public:
	USmartHelmetComponent();

	// ---------------- rig ----------------
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Rig")
	TArray<FHelmetCameraSpec> Cameras;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Rig", meta = (ClampMin = 64))
	int32 ImageWidth = 1080;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Rig", meta = (ClampMin = 64))
	int32 ImageHeight = 1920;

	/** FOV along the long (vertical) side of the portrait sensor. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Rig", meta = (ClampMin = 10, ClampMax = 175))
	float VerticalFovDeg = 141.4f;

	/** Kannala-Brandt / OpenCV fisheye coefficients k1..k4 (0 = ideal equidistant lens). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Rig")
	FVector4 LensK = FVector4(0, 0, 0, 0);

	/** Simulated installation error of each camera (std. dev., deg). Saved in rig.json only as "true_rotation". */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Rig", meta = (ClampMin = 0))
	float MountErrorStdDeg = 0.f;

	// ---------------- imperfections (real-hardware simulation) ----------------
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections")
	int32 ImperfectionSeed = 1;

	/** Std of k1 spread between lenses (k2..k4 get 1/2, 1/4, 1/8 of it). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections", meta = (ClampMin = 0))
	float LensKStd = 0.f;

	/** Each slave camera fires uniformly in [0, max] s after the master (cam0). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections", meta = (ClampMin = 0))
	float TriggerJitterMaxS = 0.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections", meta = (ClampMin = 0))
	float ExposureStdStops = 0.f;

	/** Std of log channel gains (AWB disagreement). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections", meta = (ClampMin = 0))
	float WhiteBalanceStd = 0.f;

	/** 0 = none, 1 = full cos^4 fall-off towards the image edge. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections", meta = (ClampMin = 0, ClampMax = 1))
	float VignetteStrength = 0.f;

	/** Gaussian sensor noise (8-bit units). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections", meta = (ClampMin = 0))
	float NoiseStd = 0.f;

	/** Also save an ideal world-aligned 360 cube capture from the helmet centre at the trigger instant (ground truth). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections")
	bool bCaptureReference = false;

	/** Per-camera truth; filled by RandomizeImperfections(). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Imperfections")
	TArray<FHelmetCameraFault> Faults;

	/** Draw new per-camera faults from the std/seed settings above (and MountErrorStdDeg). */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "SmartHelmet|Imperfections")
	void RandomizeImperfections();

	/** Rebuild Cameras with the default 7+1 layout. */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "SmartHelmet|Rig")
	void ResetToDefaultRig();

	// ---------------- capture ----------------
	/** Cube-face resolution used to synthesise each fisheye image. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture", meta = (ClampMin = 128, ClampMax = 4096))
	int32 FaceResolution = 1024;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture", meta = (ClampMin = 0.2))
	float CaptureInterval = 5.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture")
	bool bAutoCapture = false;

	/** Fixed exposure for all 8 cameras (same as locking AE on the real modules). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture")
	bool bLockExposure = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture", meta = (EditCondition = "bLockExposure"))
	float ExposureEV100 = 12.5f;

	/** Root folder; empty = <Project>/Saved/SmartHelmet. A new session sub-folder is created per play. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture")
	FString OutputRoot;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Capture")
	bool bShowOnScreenMessages = true;

	// ---------------- IMU ----------------
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|IMU", meta = (ClampMin = 1, ClampMax = 400))
	float ImuRateHz = 100.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|IMU")
	bool bLogImu = true;

	// ---------------- visuals ----------------
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Visual")
	bool bShowHelmetMesh = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Visual")
	bool bDrawCameraAxes = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Visual")
	FLinearColor ShellColor = FLinearColor(1.f, 0.62f, 0.02f);

	UPROPERTY(BlueprintAssignable, Category = "SmartHelmet")
	FOnHelmetFrameCaptured OnFrameCaptured;

	// ---------------- API ----------------
	/** Synchronous capture of all cameras. Returns the frame folder. */
	UFUNCTION(BlueprintCallable, Category = "SmartHelmet")
	FString CaptureFrame();

	UFUNCTION(BlueprintCallable, Category = "SmartHelmet")
	void SetAutoCapture(bool bEnabled, float IntervalSeconds = 5.f);

	/** rig.json content (intrinsics + extrinsics, Python stitcher format). */
	UFUNCTION(BlueprintCallable, Category = "SmartHelmet")
	FString GetRigJson() const;

	/** Current BNO055-like record as JSON. */
	UFUNCTION(BlueprintCallable, Category = "SmartHelmet")
	FString GetImuJson() const;

	UFUNCTION(BlueprintPure, Category = "SmartHelmet")
	FString GetSessionDir() const { return SessionDir; }

	UFUNCTION(BlueprintPure, Category = "SmartHelmet")
	int32 GetFrameCount() const { return FrameIndex; }

	UFUNCTION(BlueprintPure, Category = "SmartHelmet")
	FString GetLastFrameDir() const { return LastFrameDir; }

	/** World transform of camera i (including the simulated mounting error). */
	UFUNCTION(BlueprintPure, Category = "SmartHelmet")
	FTransform GetCameraWorldTransform(int32 Index) const;

	UFUNCTION(BlueprintPure, Category = "SmartHelmet")
	bool IsCapturePending() const { return Pending.bActive; }

	/** Find the first helmet in a world (PIE or editor). */
	static USmartHelmetComponent* FindInWorld(const UWorld* World);

	// UActorComponent
	virtual void OnRegister() override;
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& E) override;
#endif

private:
	struct FLutEntry { uint8 Face; float U; float V; float Theta; };

	struct FPendingFrame
	{
		bool bActive = false;
		double TriggerTime = 0;
		FString FrameDir;
		FString Meta;
		TArray<bool> Done;
		TArray<TArray<FColor>> Images;
		TArray<float> Delays;
	};
	FPendingFrame Pending;

	void CaptureCamera(int32 CamIdx);
	void FinalizePending();
	void CaptureReferenceCube(const FString& FrameDir, const FVector& Center);
	FVector4 CameraK(int32 CamIdx) const;

	void BuildVisuals();
	void EnsureCapture();
	void BuildLut(int32 CamIdx);
	void StartSession();
	float FocalPx() const;
	FVector PixelToCameraRay(float U, float V, const FVector4& K) const;
	void SampleImu(float Dt);
	void FlushImu();
	FString ImuRecordJson() const;
	void WriteTextFile(const FString& Path, const FString& Text) const;

	UPROPERTY(Transient)
	TObjectPtr<USceneCaptureComponent2D> Capture;

	UPROPERTY(Transient)
	TObjectPtr<UTextureRenderTarget2D> FaceTarget;

	UPROPERTY(Transient)
	TArray<TObjectPtr<UStaticMeshComponent>> VisualParts;

	TArray<TArray<FLutEntry>> Luts;          // one per camera (lenses differ)
	TArray<FVector4> LutK;
	uint8 UsedFaces = 0;
	int32 LutW = 0, LutH = 0, LutFace = 0;

	FString SessionDir;
	FString LastFrameDir;
	int32 FrameIndex = 0;
	float CaptureTimer = 0.f;

	// IMU state
	float ImuAccum = 0.f;
	bool bImuInit = false;
	FQuat PrevQuat = FQuat::Identity;
	FVector PrevLoc = FVector::ZeroVector;
	FVector PrevVel = FVector::ZeroVector;
	FVector LastGyro = FVector::ZeroVector;
	FVector LastLinAcc = FVector::ZeroVector;
	TArray<FString> ImuLines;
};
