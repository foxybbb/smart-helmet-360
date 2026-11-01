#include "SmartHelmet/SmartHelmetTestRange.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Math/RandomStream.h"
#include "UObject/ConstructorHelpers.h"

namespace
{
	UInstancedStaticMeshComponent* MakeIsm(AActor* Owner, const TCHAR* Name, const TCHAR* Mesh, const TCHAR* Mat)
	{
		UInstancedStaticMeshComponent* C = Owner->CreateDefaultSubobject<UInstancedStaticMeshComponent>(Name);
		FString MeshPath(Mesh), MatPath(Mat);
		if (UStaticMesh* M = ConstructorHelpersInternal::FindOrLoadObject<UStaticMesh>(MeshPath, LOAD_None)) { C->SetStaticMesh(M); }
		if (UMaterialInterface* Mi = ConstructorHelpersInternal::FindOrLoadObject<UMaterialInterface>(MatPath, LOAD_None)) { C->SetMaterial(0, Mi); }
		C->SetupAttachment(Owner->GetRootComponent());
		C->SetMobility(EComponentMobility::Static);
		return C;
	}
}

ASmartHelmetTestRange::ASmartHelmetTestRange()
{
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	Floor = MakeIsm(this, TEXT("Floor"), TEXT("/Game/LevelPrototyping/Meshes/SM_Cube.SM_Cube"),
		TEXT("/Game/LevelPrototyping/Materials/MI_PrototypeGrid_Gray.MI_PrototypeGrid_Gray"));
	Boxes = MakeIsm(this, TEXT("Boxes"), TEXT("/Game/LevelPrototyping/Meshes/SM_ChamferCube.SM_ChamferCube"),
		TEXT("/Game/LevelPrototyping/Materials/MI_PrototypeGrid_TopDark.MI_PrototypeGrid_TopDark"));
	Pillars = MakeIsm(this, TEXT("Pillars"), TEXT("/Game/LevelPrototyping/Meshes/SM_Cylinder.SM_Cylinder"),
		TEXT("/Game/LevelPrototyping/Materials/MI_PrototypeGrid_Gray_02.MI_PrototypeGrid_Gray_02"));
	Walls = MakeIsm(this, TEXT("Walls"), TEXT("/Game/LevelPrototyping/Meshes/SM_Cube.SM_Cube"),
		TEXT("/Game/LevelPrototyping/Materials/MI_DefaultColorway.MI_DefaultColorway"));
}

void ASmartHelmetTestRange::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	Rebuild();
}

void ASmartHelmetTestRange::Rebuild()
{
	for (UInstancedStaticMeshComponent* C : { Floor.Get(), Boxes.Get(), Pillars.Get(), Walls.Get() })
	{
		if (C) { C->ClearInstances(); }
	}
	FRandomStream Rng(Seed);
	const float R = RadiusM * 100.f, R0 = InnerRadiusM * 100.f;
	// SM_Cube / SM_ChamferCube / SM_Cylinder are 100 cm with the pivot at the bottom centre.
	Floor->AddInstance(FTransform(FRotator::ZeroRotator, FVector(0, 0, -10.f), FVector(R * 2.4f / 100.f, R * 2.4f / 100.f, 0.1f)));

	for (int32 i = 0; i < NumProps; ++i)
	{
		const float Ang = Rng.FRandRange(0.f, 2 * PI);
		const float Dist = FMath::Sqrt(Rng.FRandRange(R0 * R0, R * R)); // uniform over the annulus
		const FVector P(FMath::Cos(Ang) * Dist, FMath::Sin(Ang) * Dist, 0.f);
		const FRotator Rot(0.f, Rng.FRandRange(0.f, 360.f), 0.f);
		if (Rng.FRand() < 0.65f)
		{
			const FVector S(Rng.FRandRange(0.4f, 2.f), Rng.FRandRange(0.4f, 2.f), Rng.FRandRange(0.3f, 2.5f));
			Boxes->AddInstance(FTransform(Rot, P, S));
			if (Rng.FRand() < 0.3f) // stacked crate
			{
				Boxes->AddInstance(FTransform(Rot + FRotator(0, 20.f, 0), P + FVector(0, 0, S.Z * 100.f), S * 0.6f));
			}
		}
		else
		{
			const float D = Rng.FRandRange(0.25f, 0.8f);
			Pillars->AddInstance(FTransform(Rot, P, FVector(D, D, Rng.FRandRange(1.5f, 5.f))));
		}
	}
	if (bPerimeterWalls)
	{
		const int32 Seg = 24;
		for (int32 i = 0; i < Seg; ++i)
		{
			const float A = 2 * PI * i / Seg;
			const float Len = 2 * PI * R * 1.1f / Seg;
			const FVector P(FMath::Cos(A) * R * 1.1f, FMath::Sin(A) * R * 1.1f, 0.f);
			const float H = (i % 3 == 0) ? 8.f : 4.f;
			Walls->AddInstance(FTransform(FRotator(0, FMath::RadiansToDegrees(A) + 90.f, 0), P, FVector(Len / 100.f, 0.4f, H)));
		}
	}
}
