// Procedural test site for panorama stitching: textured boxes, pillars and walls
// at 1.5 .. Radius metres around the actor (near objects = strong parallax).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SmartHelmetTestRange.generated.h"

class UInstancedStaticMeshComponent;

UCLASS()
class MYPROJECT_API ASmartHelmetTestRange : public AActor
{
	GENERATED_BODY()

public:
	ASmartHelmetTestRange();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TestRange")
	int32 Seed = 7;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TestRange", meta = (ClampMin = 0, ClampMax = 2000))
	int32 NumProps = 140;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TestRange", meta = (ClampMin = 2))
	float RadiusM = 18.f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TestRange", meta = (ClampMin = 0.5))
	float InnerRadiusM = 1.5f;

	/** Ring of "platform" walls around the site. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TestRange")
	bool bPerimeterWalls = true;

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "TestRange")
	void Rebuild();

	virtual void OnConstruction(const FTransform& Transform) override;

private:
	UPROPERTY(VisibleAnywhere, Category = "TestRange")
	TObjectPtr<UInstancedStaticMeshComponent> Boxes;

	UPROPERTY(VisibleAnywhere, Category = "TestRange")
	TObjectPtr<UInstancedStaticMeshComponent> Pillars;

	UPROPERTY(VisibleAnywhere, Category = "TestRange")
	TObjectPtr<UInstancedStaticMeshComponent> Walls;

	UPROPERTY(VisibleAnywhere, Category = "TestRange")
	TObjectPtr<UInstancedStaticMeshComponent> Floor;
};
