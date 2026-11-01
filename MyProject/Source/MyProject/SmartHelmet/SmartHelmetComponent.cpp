#include "SmartHelmet/SmartHelmetComponent.h"

#include "Async/ParallelFor.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/StaticMeshComponent.h"
#include "DrawDebugHelpers.h"
#include "Engine/Engine.h"
#include "Engine/StaticMesh.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "ImageCore.h"
#include "ImageUtils.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Math/RandomStream.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "TextureResource.h"
#include "UObject/UObjectIterator.h"

DEFINE_LOG_CATEGORY_STATIC(LogSmartHelmet, Log, All);

namespace SmartHelmet
{
	static const FRotator FaceRot[6] = {
		FRotator(0, 0, 0),   // 0 front  (+X)
		FRotator(0, 90, 0),  // 1 right  (+Y)
		FRotator(0, -90, 0), // 2 left   (-Y)
		FRotator(90, 0, 0),  // 3 up     (+Z)
		FRotator(-90, 0, 0), // 4 down   (-Z)
		FRotator(0, 180, 0)  // 5 back   (-X)
	};

	static FString V3(const FVector& V, int32 Digits = 6)
	{
		return FString::Printf(TEXT("[%.*f, %.*f, %.*f]"), Digits, V.X, Digits, V.Y, Digits, V.Z);
	}

	static FString K4(const FVector4& K)
	{
		return FString::Printf(TEXT("[%.6f, %.6f, %.6f, %.6f]"), K.X, K.Y, K.Z, K.W);
	}

	static constexpr float RingPitch = 9.3f;
	static constexpr int32 RingCount = 7;
	static constexpr float RingRadiusCm = 10.5f;
	static constexpr float TopHeightCm = 10.f;
}

USmartHelmetComponent::USmartHelmetComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PostPhysics;
	ResetToDefaultRig();
}

void USmartHelmetComponent::ResetToDefaultRig()
{
	using namespace SmartHelmet;
	Cameras.Reset();
	for (int32 i = 0; i < RingCount; ++i)
	{
		const float Yaw = i * 360.f / RingCount;
		FHelmetCameraSpec S;
		S.Name = FString::Printf(TEXT("cam%d"), i);
		S.Position = FVector(FMath::Cos(FMath::DegreesToRadians(Yaw)), FMath::Sin(FMath::DegreesToRadians(Yaw)), 0.f) * RingRadiusCm;
		S.Rotation = FRotator(RingPitch, Yaw, 0.f);
		Cameras.Add(S);
	}
	FHelmetCameraSpec Top;
	Top.Name = TEXT("top7");
	Top.Position = FVector(0, 0, TopHeightCm);
	Top.Rotation = FRotator(90.f, 0.f, 0.f);
	Cameras.Add(Top);
	Faults.Reset();
	Luts.Reset();
}

void USmartHelmetComponent::RandomizeImperfections()
{
	FRandomStream R(ImperfectionSeed);
	auto G = [&R]()
	{
		const float U1 = FMath::Max(R.GetFraction(), 1e-6f);
		const float U2 = R.GetFraction();
		return FMath::Sqrt(-2.f * FMath::Loge(U1)) * FMath::Cos(2.f * PI * U2);
	};
	Faults.SetNum(Cameras.Num());
	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		FHelmetCameraFault& F = Faults[i];
		F.MountError = FRotator(G() * MountErrorStdDeg, G() * MountErrorStdDeg, G() * MountErrorStdDeg);
		F.LensKDelta = FVector4(G() * LensKStd, G() * LensKStd * 0.5f, G() * LensKStd * 0.25f, G() * LensKStd * 0.125f);
		F.TriggerDelay = (i == 0) ? 0.f : R.GetFraction() * TriggerJitterMaxS;
		F.ExposureGain = FMath::Pow(2.f, G() * ExposureStdStops);
		F.WhiteBalance = FLinearColor(FMath::Exp(G() * WhiteBalanceStd), 1.f, FMath::Exp(G() * WhiteBalanceStd), 1.f);
	}
	Luts.Reset();
	UE_LOG(LogSmartHelmet, Log, TEXT("Imperfections randomized (seed %d): mount %.2f deg, k %.3f, jitter %.2f s, exposure %.2f stops"),
		ImperfectionSeed, MountErrorStdDeg, LensKStd, TriggerJitterMaxS, ExposureStdStops);
}

FVector4 USmartHelmetComponent::CameraK(int32 CamIdx) const
{
	FVector4 K = LensK;
	if (Faults.IsValidIndex(CamIdx))
	{
		K += Faults[CamIdx].LensKDelta;
	}
	return K;
}

float USmartHelmetComponent::FocalPx() const
{
	return (ImageHeight * 0.5f) / FMath::DegreesToRadians(VerticalFovDeg * 0.5f);
}

FVector USmartHelmetComponent::PixelToCameraRay(float U, float V, const FVector4& K) const
{
	const float F = FocalPx();
	const float Mx = (U - (ImageWidth - 1) * 0.5f) / F;
	const float My = (V - (ImageHeight - 1) * 0.5f) / F;
	const float ThetaD = FMath::Sqrt(Mx * Mx + My * My);
	float Theta = ThetaD;
	if (K.X != 0 || K.Y != 0 || K.Z != 0 || K.W != 0)
	{
		for (int32 It = 0; It < 10; ++It)
		{
			const float T2 = Theta * Theta;
			const float Fv = Theta * (1 + T2 * (K.X + T2 * (K.Y + T2 * (K.Z + T2 * K.W)))) - ThetaD;
			const float Df = 1 + T2 * (3 * K.X + T2 * (5 * K.Y + T2 * (7 * K.Z + 9 * K.W * T2)));
			Theta -= Fv / FMath::Max(Df, 0.1f);
		}
	}
	const float S = ThetaD > 1e-9f ? FMath::Sin(Theta) / ThetaD : 0.f;
	// OpenCV camera (x right, y down, z fwd) -> UE local (X fwd, Y right, Z up)
	return FVector(FMath::Cos(Theta), Mx * S, -My * S);
}

void USmartHelmetComponent::BuildLut(int32 CamIdx)
{
	using namespace SmartHelmet;
	const FVector4 K = CameraK(CamIdx);
	TArray<FLutEntry>& Lut = Luts[CamIdx];
	Lut.SetNumUninitialized(LutW * LutH);

	FVector Fwd[6], Right[6], Up[6];
	for (int32 f = 0; f < 6; ++f)
	{
		const FRotationMatrix M(FaceRot[f]);
		Fwd[f] = M.GetUnitAxis(EAxis::X);
		Right[f] = M.GetUnitAxis(EAxis::Y);
		Up[f] = M.GetUnitAxis(EAxis::Z);
	}
	const float N = (float)FaceResolution;
	ParallelFor(LutH, [&](int32 Y)
	{
		for (int32 X = 0; X < LutW; ++X)
		{
			const FVector R = PixelToCameraRay((float)X, (float)Y, K);
			int32 Best = 0;
			float BestDot = -2.f;
			for (int32 f = 0; f < 6; ++f)
			{
				const float D = FVector::DotProduct(R, Fwd[f]);
				if (D > BestDot) { BestDot = D; Best = f; }
			}
			FLutEntry& E = Lut[Y * LutW + X];
			E.Face = (uint8)Best;
			E.U = (FVector::DotProduct(R, Right[Best]) / BestDot + 1.f) * 0.5f * N - 0.5f;
			E.V = (1.f - FVector::DotProduct(R, Up[Best]) / BestDot) * 0.5f * N - 0.5f;
			E.Theta = FMath::Acos(FMath::Clamp((float)R.X, -1.f, 1.f));
		}
	});
	for (const FLutEntry& E : Lut) { UsedFaces |= (1 << E.Face); }
	if (LutK.Num() != Cameras.Num()) { LutK.SetNum(Cameras.Num()); }
	LutK[CamIdx] = K;
}

void USmartHelmetComponent::OnRegister()
{
	Super::OnRegister();
	BuildVisuals();
}

void USmartHelmetComponent::BuildVisuals()
{
	for (UStaticMeshComponent* P : VisualParts)
	{
		if (P) { P->DestroyComponent(); }
	}
	VisualParts.Reset();
	AActor* Owner = GetOwner();
	if (!bShowHelmetMesh || !Owner) { return; }

	UStaticMesh* Sphere = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	UStaticMesh* Cylinder = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
	UMaterialInterface* BaseMat = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
	if (!Sphere || !Cylinder) { return; }

	auto MakePart = [&](UStaticMesh* Mesh, const FTransform& Rel, const FLinearColor& Color)
	{
		UStaticMeshComponent* C = NewObject<UStaticMeshComponent>(Owner, NAME_None, RF_Transient | RF_TextExportTransient);
		C->SetStaticMesh(Mesh);
		C->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		C->SetCastShadow(true);
		C->SetupAttachment(this);
		C->SetRelativeTransform(Rel);
		if (BaseMat)
		{
			UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMat, C);
			MID->SetVectorParameterValue(TEXT("Color"), Color);
			C->SetMaterial(0, MID);
		}
		C->RegisterComponent();
		VisualParts.Add(C);
		return C;
	};

	MakePart(Sphere, FTransform(FRotator::ZeroRotator, FVector(1.5f, 0, -1.f), FVector(0.25f, 0.215f, 0.22f)), ShellColor);
	MakePart(Cylinder, FTransform(FRotator::ZeroRotator, FVector(3.f, 0, -2.5f), FVector(0.30f, 0.25f, 0.012f)), ShellColor * 0.85f);
	const FLinearColor Housing(0.03f, 0.03f, 0.035f);
	const FLinearColor Lens(0.f, 0.08f, 0.25f);
	for (const FHelmetCameraSpec& S : Cameras)
	{
		const FQuat Q = S.Rotation.Quaternion() * FRotator(-90.f, 0.f, 0.f).Quaternion();
		const FVector Axis = S.Rotation.Vector();
		MakePart(Cylinder, FTransform(Q, S.Position - Axis * 1.f, FVector(0.03f, 0.03f, 0.03f)), Housing);
		MakePart(Sphere, FTransform(Q, S.Position + Axis * 0.6f, FVector(0.018f)), Lens);
	}
	if (Capture)
	{
		for (UStaticMeshComponent* P : VisualParts) { Capture->HideComponent(P); }
	}
}

#if WITH_EDITOR
void USmartHelmetComponent::PostEditChangeProperty(FPropertyChangedEvent& E)
{
	Super::PostEditChangeProperty(E);
	Luts.Reset();
	BuildVisuals();
}
#endif

void USmartHelmetComponent::EnsureCapture()
{
	if (!FaceTarget || FaceTarget->SizeX != FaceResolution)
	{
		FaceTarget = NewObject<UTextureRenderTarget2D>(this, NAME_None, RF_Transient);
		FaceTarget->ClearColor = FLinearColor::Black;
		FaceTarget->InitCustomFormat(FaceResolution, FaceResolution, PF_B8G8R8A8, /*bForceLinearGamma*/ true);
		FaceTarget->UpdateResourceImmediate(true);
	}
	if (!Capture)
	{
		Capture = NewObject<USceneCaptureComponent2D>(GetOwner() ? (UObject*)GetOwner() : (UObject*)this, NAME_None, RF_Transient);
		Capture->bCaptureEveryFrame = false;
		Capture->bCaptureOnMovement = false;
		Capture->bAlwaysPersistRenderingState = true;
		Capture->CaptureSource = SCS_FinalColorLDR;
		Capture->FOVAngle = 90.f;
		Capture->ShowFlags.SetMotionBlur(false);
		Capture->SetupAttachment(this);
		Capture->RegisterComponent();
		for (UStaticMeshComponent* P : VisualParts) { Capture->HideComponent(P); }
	}
	Capture->TextureTarget = FaceTarget;
	FPostProcessSettings& PP = Capture->PostProcessSettings;
	PP.bOverride_MotionBlurAmount = true;
	PP.MotionBlurAmount = 0.f;
	PP.bOverride_AutoExposureMinBrightness = PP.bOverride_AutoExposureMaxBrightness = bLockExposure;
	PP.AutoExposureMinBrightness = PP.AutoExposureMaxBrightness = ExposureEV100;

	if (Faults.Num() != Cameras.Num())
	{
		RandomizeImperfections();
	}
	const bool bSizeChanged = LutW != ImageWidth || LutH != ImageHeight || LutFace != FaceResolution;
	if (bSizeChanged || Luts.Num() != Cameras.Num())
	{
		Luts.Reset();
		Luts.SetNum(Cameras.Num());
		LutK.Reset();
		LutW = ImageWidth; LutH = ImageHeight; LutFace = FaceResolution;
		UsedFaces = 0;
	}
	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		if (Luts[i].Num() != LutW * LutH || !LutK.IsValidIndex(i) || LutK[i] != CameraK(i))
		{
			BuildLut(i);
		}
	}
}

FTransform USmartHelmetComponent::GetCameraWorldTransform(int32 Index) const
{
	if (!Cameras.IsValidIndex(Index)) { return GetComponentTransform(); }
	const FHelmetCameraSpec& S = Cameras[Index];
	const FQuat Err = Faults.IsValidIndex(Index) ? Faults[Index].MountError.Quaternion() : FQuat::Identity;
	const FTransform Rel(S.Rotation.Quaternion() * Err, S.Position);
	FTransform W = Rel * GetComponentTransform();
	W.SetScale3D(FVector::OneVector);
	return W;
}

void USmartHelmetComponent::WriteTextFile(const FString& Path, const FString& Text) const
{
	FFileHelper::SaveStringToFile(Text, *Path, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}

void USmartHelmetComponent::StartSession()
{
	FString Root = OutputRoot.IsEmpty() ? FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("SmartHelmet")) : OutputRoot;
	Root = FPaths::ConvertRelativePathToFull(Root);
	const UWorld* W = GetWorld();
	const FString Tag = W && W->IsPlayInEditor() ? TEXT("PIE") : (W && W->WorldType == EWorldType::Editor ? TEXT("Editor") : TEXT("Game"));
	SessionDir = FPaths::Combine(Root, FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S_")) + Tag);
	IFileManager::Get().MakeDirectory(*FPaths::Combine(SessionDir, TEXT("frames")), true);
	FrameIndex = 0;
	WriteTextFile(FPaths::Combine(SessionDir, TEXT("rig.json")), GetRigJson());
	UE_LOG(LogSmartHelmet, Log, TEXT("Smart Helmet session: %s"), *SessionDir);
}

void USmartHelmetComponent::BeginPlay()
{
	Super::BeginPlay();
	StartSession();
	CaptureTimer = 0.f;
	bImuInit = false;
}

void USmartHelmetComponent::EndPlay(const EEndPlayReason::Type Reason)
{
	if (Pending.bActive)
	{
		for (int32 i = 0; i < Pending.Done.Num(); ++i)
		{
			if (!Pending.Done[i]) { CaptureCamera(i); }
		}
		FinalizePending();
	}
	FlushImu();
	Super::EndPlay(Reason);
}

void USmartHelmetComponent::SetAutoCapture(bool bEnabled, float IntervalSeconds)
{
	bAutoCapture = bEnabled;
	CaptureInterval = FMath::Max(0.2f, IntervalSeconds);
	CaptureTimer = 0.f;
}

void USmartHelmetComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	SampleImu(DeltaTime);
	if (Pending.bActive && GetWorld())
	{
		const double Elapsed = GetWorld()->GetTimeSeconds() - Pending.TriggerTime;
		bool bAll = true;
		for (int32 i = 0; i < Pending.Done.Num(); ++i)
		{
			if (!Pending.Done[i] && Elapsed >= Pending.Delays[i]) { CaptureCamera(i); }
			bAll &= Pending.Done[i];
		}
		if (bAll) { FinalizePending(); }
	}
	if (bAutoCapture && !Pending.bActive)
	{
		CaptureTimer += DeltaTime;
		if (CaptureTimer >= CaptureInterval)
		{
			CaptureTimer = 0.f;
			CaptureFrame();
		}
	}
	if (bDrawCameraAxes)
	{
		for (int32 i = 0; i < Cameras.Num(); ++i)
		{
			const FTransform T = GetCameraWorldTransform(i);
			DrawDebugDirectionalArrow(GetWorld(), T.GetLocation(), T.GetLocation() + T.GetRotation().GetForwardVector() * 40.f,
				6.f, i == 7 ? FColor::Cyan : FColor::Green, false, -1.f, 0, 0.6f);
		}
	}
}

// ---------------------------------------------------------------- IMU (BNO055 record format, report table 4)
void USmartHelmetComponent::SampleImu(float Dt)
{
	if (Dt <= 0.f) { return; }
	const FQuat Q = GetComponentQuat();
	const FVector L = GetComponentLocation() / 100.0; // m
	if (!bImuInit)
	{
		PrevQuat = Q; PrevLoc = L; PrevVel = FVector::ZeroVector; bImuInit = true;
		return;
	}
	const FVector Vel = (L - PrevLoc) / Dt;
	const FVector AccW = (Vel - PrevVel) / Dt;
	FQuat dQ = Q * PrevQuat.Inverse();
	dQ.EnforceShortestArcWith(FQuat::Identity);
	FVector Axis; float Angle;
	dQ.ToAxisAndAngle(Axis, Angle);
	LastGyro = Q.UnrotateVector(Axis * (Angle / Dt));
	LastLinAcc = Q.UnrotateVector(AccW);
	PrevQuat = Q; PrevLoc = L; PrevVel = Vel;

	ImuAccum += Dt;
	const float Period = 1.f / ImuRateHz;
	if (bLogImu && !SessionDir.IsEmpty() && ImuAccum >= Period)
	{
		ImuAccum = FMath::Fmod(ImuAccum, Period);
		ImuLines.Add(ImuRecordJson());
		if (ImuLines.Num() >= 200) { FlushImu(); }
	}
}

void USmartHelmetComponent::FlushImu()
{
	if (ImuLines.Num() == 0 || SessionDir.IsEmpty()) { return; }
	const FString Text = FString::Join(ImuLines, TEXT("\n")) + TEXT("\n");
	FFileHelper::SaveStringToFile(Text, *FPaths::Combine(SessionDir, TEXT("imu.jsonl")),
		FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append);
	ImuLines.Reset();
}

FString USmartHelmetComponent::ImuRecordJson() const
{
	using SmartHelmet::V3;
	const FQuat Q = GetComponentQuat();
	const FVector G(0, 0, 9.80665);
	const FVector Grav = Q.UnrotateVector(G);
	const FVector Acc = LastLinAcc + Grav;
	const FVector Mag = Q.UnrotateVector(FVector(20.0, 0.0, -45.0));
	const FRotator R = Q.Rotator();
	return FString::Printf(
		TEXT("{\"timestamp\": \"%s\", \"temperature_C\": 25.0, \"acceleration_m_s2\": %s, \"magnetic_uT\": %s, "
			 "\"gyro_rad_s\": %s, \"euler_deg\": [%.4f, %.4f, %.4f], \"quaternion\": [%.7f, %.7f, %.7f, %.7f], "
			 "\"linear_acceleration_m_s2\": %s, \"gravity_m_s2\": %s}"),
		*FDateTime::UtcNow().ToIso8601(), *V3(Acc, 4), *V3(Mag, 3), *V3(LastGyro, 5),
		R.Roll, R.Pitch, R.Yaw, Q.W, Q.X, Q.Y, Q.Z, *V3(LastLinAcc, 4), *V3(Grav, 4));
}

FString USmartHelmetComponent::GetImuJson() const { return ImuRecordJson(); }

// ---------------------------------------------------------------- rig.json (nominal = what the stitcher knows)
FString USmartHelmetComponent::GetRigJson() const
{
	using namespace SmartHelmet;
	const float F = FocalPx();
	TArray<FString> Cams;
	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		const FHelmetCameraSpec& S = Cameras[i];
		const FHelmetCameraFault Fl = Faults.IsValidIndex(i) ? Faults[i] : FHelmetCameraFault();
		const FRotator True = (S.Rotation.Quaternion() * Fl.MountError.Quaternion()).Rotator();
		Cams.Add(FString::Printf(
			TEXT("    {\"name\": \"%s\", \"width\": %d, \"height\": %d, \"fx\": %.6f, \"fy\": %.6f, \"cx\": %.3f, \"cy\": %.3f, "
				 "\"k\": %s, \"position_m\": %s, \"rotation_pyr_deg\": [%.6f, %.6f, %.6f], "
				 "\"true_rotation_pyr_deg\": [%.6f, %.6f, %.6f], \"true_k\": %s, \"true_trigger_delay_s\": %.4f, "
				 "\"true_exposure_gain\": %.4f, \"true_white_balance\": [%.4f, %.4f, %.4f], \"max_theta_deg\": 89.0}"),
			*S.Name, ImageWidth, ImageHeight, F, F, (ImageWidth - 1) * 0.5, (ImageHeight - 1) * 0.5,
			*K4(LensK), *V3(S.Position / 100.0), S.Rotation.Pitch, S.Rotation.Yaw, S.Rotation.Roll,
			True.Pitch, True.Yaw, True.Roll, *K4(CameraK(i)), Fl.TriggerDelay, Fl.ExposureGain,
			Fl.WhiteBalance.R, Fl.WhiteBalance.G, Fl.WhiteBalance.B));
	}
	return FString::Printf(TEXT("{\n  \"name\": \"SmartHelmet_8cam_UE\",\n  \"imu_from_rig_pyr_deg\": [0, 0, 0],\n  \"source\": \"UnrealEngine\",\n")
		TEXT("  \"vfov_deg\": %.3f, \"face_resolution\": %d,\n  \"imperfections\": {\"seed\": %d, \"mount_error_std_deg\": %.3f, \"lens_k_std\": %.4f, ")
		TEXT("\"trigger_jitter_max_s\": %.3f, \"exposure_std_stops\": %.3f, \"white_balance_std\": %.3f, \"vignette\": %.3f, \"noise_std\": %.2f},\n")
		TEXT("  \"cameras\": [\n%s\n  ]\n}\n"),
		VerticalFovDeg, FaceResolution, ImperfectionSeed, MountErrorStdDeg, LensKStd, TriggerJitterMaxS, ExposureStdStops,
		WhiteBalanceStd, VignetteStrength, NoiseStd, *FString::Join(Cams, TEXT(",\n")));
}

// ---------------------------------------------------------------- capture
void USmartHelmetComponent::CaptureCamera(int32 CamIdx)
{
	const int32 N = FaceResolution;
	const int32 W = ImageWidth, H = ImageHeight;
	TArray<TArray<FColor>> Faces;
	Faces.SetNum(6);
	FTextureRenderTargetResource* Res = FaceTarget->GameThread_GetRenderTargetResource();
	const FTransform CamT = GetCameraWorldTransform(CamIdx);
	for (int32 f = 0; f < 6; ++f)
	{
		if (!(UsedFaces & (1 << f))) { continue; }
		Capture->SetWorldLocationAndRotation(CamT.GetLocation(), CamT.GetRotation() * SmartHelmet::FaceRot[f].Quaternion());
		Capture->CaptureScene();
		Res->ReadPixels(Faces[f]);
	}

	// sensor / ISP imperfections
	const FHelmetCameraFault Fl = Faults.IsValidIndex(CamIdx) ? Faults[CamIdx] : FHelmetCameraFault();
	const float Gain[3] = { Fl.ExposureGain * Fl.WhiteBalance.R, Fl.ExposureGain * Fl.WhiteBalance.G, Fl.ExposureGain * Fl.WhiteBalance.B };
	const bool bFx = FMath::Abs(Gain[0] - 1) > 1e-4f || FMath::Abs(Gain[1] - 1) > 1e-4f || FMath::Abs(Gain[2] - 1) > 1e-4f
		|| VignetteStrength > 0 || NoiseStd > 0;
	float ToLin[256];
	for (int32 v = 0; v < 256; ++v) { ToLin[v] = FMath::Pow(v / 255.f, 2.2f); }
	const TArray<FLutEntry>& Lut = Luts[CamIdx];
	const int32 Seed = ImperfectionSeed * 7919 + FrameIndex * 131 + CamIdx * 17;

	TArray<FColor>& Img = Pending.Images[CamIdx];
	Img.SetNumUninitialized(W * H);
	ParallelFor(H, [&](int32 Y)
	{
		FRandomStream Rs(Seed + Y);
		for (int32 X = 0; X < W; ++X)
		{
			const FLutEntry& E = Lut[Y * W + X];
			const TArray<FColor>& Src = Faces[E.Face];
			if (Src.Num() != N * N) { Img[Y * W + X] = FColor::Black; continue; }
			const float U = FMath::Clamp(E.U, 0.f, N - 1.001f);
			const float V = FMath::Clamp(E.V, 0.f, N - 1.001f);
			const int32 X0 = (int32)U, Y0 = (int32)V;
			const float Fx = U - X0, Fy = V - Y0;
			const FColor& C00 = Src[Y0 * N + X0];
			const FColor& C10 = Src[Y0 * N + X0 + 1];
			const FColor& C01 = Src[(Y0 + 1) * N + X0];
			const FColor& C11 = Src[(Y0 + 1) * N + X0 + 1];
			auto Lerp2 = [&](uint8 a, uint8 b, uint8 c, uint8 d)
			{
				const float Top = a + (b - a) * Fx;
				const float Bot = c + (d - c) * Fx;
				return Top + (Bot - Top) * Fy;
			};
			float Ch[3] = { Lerp2(C00.R, C10.R, C01.R, C11.R), Lerp2(C00.G, C10.G, C01.G, C11.G), Lerp2(C00.B, C10.B, C01.B, C11.B) };
			if (bFx)
			{
				const float Ct = FMath::Cos(FMath::Min(E.Theta, 1.5f));
				const float Vig = 1.f - VignetteStrength * (1.f - Ct * Ct * Ct * Ct);
				for (int32 c = 0; c < 3; ++c)
				{
					const int32 I = FMath::Clamp((int32)Ch[c], 0, 254);
					const float Lin = FMath::Lerp(ToLin[I], ToLin[I + 1], Ch[c] - I) * Gain[c] * Vig;
					Ch[c] = FMath::Pow(FMath::Clamp(Lin, 0.f, 1.f), 1.f / 2.2f) * 255.f
						+ (NoiseStd > 0 ? Rs.FRandRange(-1.732f, 1.732f) * NoiseStd : 0.f);
				}
			}
			Img[Y * W + X] = FColor((uint8)FMath::Clamp(FMath::RoundToInt(Ch[0]), 0, 255),
				(uint8)FMath::Clamp(FMath::RoundToInt(Ch[1]), 0, 255), (uint8)FMath::Clamp(FMath::RoundToInt(Ch[2]), 0, 255), 255);
		}
	});
	Pending.Done[CamIdx] = true;
}

void USmartHelmetComponent::CaptureReferenceCube(const FString& FrameDir, const FVector& Center)
{
	FTextureRenderTargetResource* Res = FaceTarget->GameThread_GetRenderTargetResource();
	TArray<TArray<FColor>> Faces;
	Faces.SetNum(6);
	for (int32 f = 0; f < 6; ++f)
	{
		Capture->SetWorldLocationAndRotation(Center, SmartHelmet::FaceRot[f].Quaternion());   // world aligned
		Capture->CaptureScene();
		Res->ReadPixels(Faces[f]);
		for (FColor& C : Faces[f]) { C.A = 255; }
	}
	const int32 N = FaceResolution;
	ParallelFor(6, [&](int32 f)
	{
		FImageUtils::SaveImageByExtension(*FPaths::Combine(FrameDir, FString::Printf(TEXT("ref_face%d.png"), f)),
			FImageView(Faces[f].GetData(), N, N));
	});
}

FString USmartHelmetComponent::CaptureFrame()
{
	UWorld* World = GetWorld();
	if (!World || Cameras.Num() == 0) { return FString(); }
	if (Pending.bActive) { return Pending.FrameDir; }
	EnsureCapture();
	if (SessionDir.IsEmpty()) { StartSession(); }

	++FrameIndex;
	const FString FrameDir = FPaths::Combine(SessionDir, TEXT("frames"), FString::Printf(TEXT("%06d"), FrameIndex));
	IFileManager::Get().MakeDirectory(*FrameDir, true);

	// Trigger instant: helmet pose + IMU (what the master Pi logs)
	const FTransform HelmetT = GetComponentTransform();
	const FQuat HQ = HelmetT.GetRotation();
	const bool bTicking = World->IsGameWorld();

	Pending = FPendingFrame();
	Pending.bActive = true;
	Pending.FrameDir = FrameDir;
	Pending.TriggerTime = World->GetTimeSeconds();
	Pending.Images.SetNum(Cameras.Num());
	Pending.Done.Init(false, Cameras.Num());
	Pending.Delays.SetNum(Cameras.Num());
	TArray<FString> Truth;
	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		Pending.Delays[i] = (bTicking && Faults.IsValidIndex(i)) ? Faults[i].TriggerDelay : 0.f;
		Truth.Add(FString::Printf(TEXT("{\"name\": \"%s\", \"trigger_delay_s\": %.4f}"), *Cameras[i].Name, Pending.Delays[i]));
	}

	using SmartHelmet::V3;
	const FVector Ax = HQ.GetAxisX(), Ay = HQ.GetAxisY(), Az = HQ.GetAxisZ();
	Pending.Meta = FString::Printf(
		TEXT("{\n  \"frame\": %d,\n  \"timestamp\": \"%s\",\n  \"world\": \"%s\",\n  \"helmet_location_m\": %s,\n")
		TEXT("  \"helmet_rotation_pyr_deg\": [%.5f, %.5f, %.5f],\n")
		TEXT("  \"rig_to_world\": [[%.7f, %.7f, %.7f], [%.7f, %.7f, %.7f], [%.7f, %.7f, %.7f]],\n")
		TEXT("  \"exposure_ev100\": %.2f,\n  \"reference_cube\": %s,\n  \"truth\": [%s],\n  \"imu\": %s\n}\n"),
		FrameIndex, *FDateTime::UtcNow().ToIso8601(), *World->GetMapName(), *V3(HelmetT.GetLocation() / 100.0),
		HQ.Rotator().Pitch, HQ.Rotator().Yaw, HQ.Rotator().Roll,
		Ax.X, Ay.X, Az.X, Ax.Y, Ay.Y, Az.Y, Ax.Z, Ay.Z, Az.Z,
		bLockExposure ? ExposureEV100 : -1.f, bCaptureReference ? TEXT("true") : TEXT("false"),
		*FString::Join(Truth, TEXT(", ")), *ImuRecordJson());

	if (bCaptureReference)
	{
		CaptureReferenceCube(FrameDir, HelmetT.GetLocation());
	}
	bool bAll = true;
	for (int32 i = 0; i < Cameras.Num(); ++i)
	{
		if (Pending.Delays[i] <= 0.f) { CaptureCamera(i); }
		bAll &= Pending.Done[i];
	}
	if (bAll) { FinalizePending(); }
	return FrameDir;
}

void USmartHelmetComponent::FinalizePending()
{
	if (!Pending.bActive) { return; }
	const int32 W = ImageWidth, H = ImageHeight;
	const double T0 = FPlatformTime::Seconds();
	ParallelFor(Pending.Images.Num(), [&](int32 i)
	{
		const FString Path = FPaths::Combine(Pending.FrameDir, Cameras[i].Name + TEXT(".png"));
		FImageUtils::SaveImageByExtension(*Path, FImageView(Pending.Images[i].GetData(), W, H));
	});
	WriteTextFile(FPaths::Combine(Pending.FrameDir, TEXT("meta.json")), Pending.Meta);   // written last = frame complete
	FlushImu();
	LastFrameDir = Pending.FrameDir;
	Pending.bActive = false;
	UE_LOG(LogSmartHelmet, Log, TEXT("Frame %d complete (%d cams) -> %s"), FrameIndex, Cameras.Num(), *LastFrameDir);
	if (bShowOnScreenMessages && GEngine)
	{
		GEngine->AddOnScreenDebugMessage(9001, 4.f, FColor::Green,
			FString::Printf(TEXT("SmartHelmet: photo #%d saved on %d cameras + IMU"), FrameIndex, Cameras.Num()));
	}
	OnFrameCaptured.Broadcast(FrameIndex, LastFrameDir);
	Pending.Images.Reset();
}

USmartHelmetComponent* USmartHelmetComponent::FindInWorld(const UWorld* World)
{
	if (!World) { return nullptr; }
	for (TObjectIterator<USmartHelmetComponent> It; It; ++It)
	{
		if (It->GetWorld() == World && !It->IsTemplate() && IsValid(*It) && It->IsRegistered())
		{
			return *It;
		}
	}
	return nullptr;
}
