#include "SmartHelmetToolset.h"

#include "Editor.h"
#include "Engine/World.h"
#include "HAL/PlatformProcess.h"
#include "Misc/Paths.h"
#include "SmartHelmet/SmartHelmetCharacter.h"
#include "SmartHelmet/SmartHelmetComponent.h"

namespace
{
	UWorld* HelmetWorld()
	{
		if (!GEditor) { return nullptr; }
		if (GEditor->PlayWorld) { return GEditor->PlayWorld; }
		return GEditor->GetEditorWorldContext().World();
	}

	USmartHelmetComponent* FindHelmet(FString& OutError)
	{
		UWorld* W = HelmetWorld();
		USmartHelmetComponent* H = USmartHelmetComponent::FindInWorld(W);
		if (!H)
		{
			OutError = FString::Printf(TEXT("{\"error\": \"No SmartHelmetComponent in world '%s'. Start PIE with SmartHelmetGameMode or place an ASmartHelmetCharacter in the level.\"}"),
				W ? *W->GetMapName() : TEXT("none"));
		}
		return H;
	}

	FString Esc(const FString& S) { return S.ReplaceCharWithEscapedChar(); }

	FString RepoRoot() { return FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectDir(), TEXT(".."))); }
}

FString USmartHelmetToolset::CaptureFrame()
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	if (!H) { return Err; }
	const FString Dir = H->CaptureFrame();
	if (Dir.IsEmpty()) { return TEXT("{\"error\": \"capture failed\"}"); }
	return FString::Printf(TEXT("{\"frame_dir\": \"%s\", \"session_dir\": \"%s\", \"frame\": %d, \"world\": \"%s\", \"pending\": %s, \"imu\": %s}"),
		*Esc(Dir), *Esc(H->GetSessionDir()), H->GetFrameCount(), *H->GetWorld()->GetMapName(),
		H->IsCapturePending() ? TEXT("true") : TEXT("false"), *H->GetImuJson());
}

FString USmartHelmetToolset::GetRigCalibration()
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	return H ? H->GetRigJson() : Err;
}

FString USmartHelmetToolset::SetAutoCapture(bool bEnabled, float IntervalSeconds)
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	if (!H) { return Err; }
	H->SetAutoCapture(bEnabled, IntervalSeconds > 0 ? IntervalSeconds : 5.f);
	return FString::Printf(TEXT("{\"auto_capture\": %s, \"interval_s\": %.2f, \"note\": \"periodic capture runs only while the world ticks (PIE/game)\"}"),
		H->bAutoCapture ? TEXT("true") : TEXT("false"), H->CaptureInterval);
}

FString USmartHelmetToolset::ConfigureCapture(int32 FaceResolution, float ExposureEV100, float MountErrorStdDeg)
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	if (!H) { return Err; }
	if (FaceResolution > 0) { H->FaceResolution = FMath::Clamp(FaceResolution, 256, 4096); }
	H->ExposureEV100 = ExposureEV100;
	H->bLockExposure = true;
	H->MountErrorStdDeg = FMath::Max(0.f, MountErrorStdDeg);
	return FString::Printf(TEXT("{\"face_resolution\": %d, \"exposure_ev100\": %.2f, \"mount_error_std_deg\": %.2f}"),
		H->FaceResolution, H->ExposureEV100, H->MountErrorStdDeg);
}

FString USmartHelmetToolset::ConfigureImperfections(int32 Seed, float MountErrorStdDeg, float LensKStd, float TriggerJitterMaxS,
	float ExposureStdStops, float WhiteBalanceStd, float VignetteStrength, float NoiseStd, bool bCaptureReference)
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	if (!H) { return Err; }
	H->ImperfectionSeed = Seed;
	H->MountErrorStdDeg = FMath::Max(0.f, MountErrorStdDeg);
	H->LensKStd = FMath::Max(0.f, LensKStd);
	H->TriggerJitterMaxS = FMath::Max(0.f, TriggerJitterMaxS);
	H->ExposureStdStops = FMath::Max(0.f, ExposureStdStops);
	H->WhiteBalanceStd = FMath::Max(0.f, WhiteBalanceStd);
	H->VignetteStrength = FMath::Clamp(VignetteStrength, 0.f, 1.f);
	H->NoiseStd = FMath::Max(0.f, NoiseStd);
	H->bCaptureReference = bCaptureReference;
	H->RandomizeImperfections();
	return H->GetRigJson();
}

FString USmartHelmetToolset::SetOperatorMotion(bool bEnabled, float WalkInput, float TurnDegPerSec)
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	if (!H) { return Err; }
	ASmartHelmetCharacter* C = Cast<ASmartHelmetCharacter>(H->GetOwner());
	if (!C) { return TEXT("{\"error\": \"helmet owner is not ASmartHelmetCharacter\"}"); }
	C->bAutoWalk = bEnabled;
	C->AutoWalkInput = FMath::Clamp(WalkInput, 0.f, 1.f);
	C->AutoTurnDegPerSec = TurnDegPerSec;
	return FString::Printf(TEXT("{\"auto_walk\": %s, \"input\": %.2f, \"turn_deg_s\": %.1f, \"pie\": %s}"),
		bEnabled ? TEXT("true") : TEXT("false"), C->AutoWalkInput, TurnDegPerSec,
		(GEditor && GEditor->PlayWorld) ? TEXT("true") : TEXT("false"));
}

FString USmartHelmetToolset::GetStatus()
{
	FString Err;
	USmartHelmetComponent* H = FindHelmet(Err);
	if (!H) { return Err; }
	const FTransform T = H->GetComponentTransform();
	return FString::Printf(
		TEXT("{\"world\": \"%s\", \"pie\": %s, \"session_dir\": \"%s\", \"frames\": %d, \"last_frame_dir\": \"%s\", \"auto_capture\": %s, ")
		TEXT("\"interval_s\": %.2f, \"cameras\": %d, \"helmet_location_cm\": [%.1f, %.1f, %.1f], \"imu\": %s}"),
		*H->GetWorld()->GetMapName(), (GEditor && GEditor->PlayWorld) ? TEXT("true") : TEXT("false"),
		*Esc(H->GetSessionDir()), H->GetFrameCount(), *Esc(H->GetLastFrameDir()), H->bAutoCapture ? TEXT("true") : TEXT("false"),
		H->CaptureInterval, H->Cameras.Num(), T.GetLocation().X, T.GetLocation().Y, T.GetLocation().Z, *H->GetImuJson());
}

FString USmartHelmetToolset::StitchFrame(const FString& FrameDir, int32 Width)
{
	FString Dir = FrameDir;
	if (Dir.IsEmpty())
	{
		FString Err;
		if (USmartHelmetComponent* H = FindHelmet(Err)) { Dir = H->GetLastFrameDir(); }
	}
	if (Dir.IsEmpty() || !FPaths::DirectoryExists(Dir)) { return TEXT("{\"error\": \"frame folder not found; capture a frame first\"}"); }

	const FString Python = FPaths::Combine(RepoRoot(), TEXT(".venv/Scripts/python.exe"));
	const FString Tools = FPaths::Combine(RepoRoot(), TEXT("Tools"));
	if (!FPaths::FileExists(Python)) { return FString::Printf(TEXT("{\"error\": \"python venv not found: %s\"}"), *Esc(Python)); }

	const FString Args = FString::Printf(TEXT("-m smart_helmet_pano stitch \"%s\" --width %d"), *Dir, Width > 0 ? Width : 2048);
	int32 Code = -1;
	FString Out, StdErr;
	FPlatformProcess::ExecProcess(*Python, *Args, &Code, &Out, &StdErr, *Tools);
	if (Code != 0) { return FString::Printf(TEXT("{\"error\": \"stitcher exit %d\", \"stderr\": \"%s\"}"), Code, *Esc(StdErr.Right(1500))); }
	Out.TrimStartAndEndInline();
	TArray<FString> Lines;
	Out.ParseIntoArrayLines(Lines);
	return Lines.Num() ? Lines.Last() : Out;
}

FToolsetImage USmartHelmetToolset::ViewImage(const FString& Path)
{
	FToolsetImage Img;
	Img.SetFromFile(Path);
	return Img;
}
