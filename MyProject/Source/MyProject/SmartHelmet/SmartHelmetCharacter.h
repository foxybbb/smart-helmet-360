// Operator character (UE5 Manny mannequin) wearing the Smart Helmet.
// Controls: WASD/mouse (template input), Space jump, C capture, T toggle auto-capture, V helmet/3rd-person view.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "SmartHelmetCharacter.generated.h"

class USpringArmComponent;
class UCameraComponent;
class UInputAction;
class UInputMappingContext;
class USmartHelmetComponent;
struct FInputActionValue;

UCLASS()
class MYPROJECT_API ASmartHelmetCharacter : public ACharacter
{
	GENERATED_BODY()

public:
	ASmartHelmetCharacter();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<USpringArmComponent> CameraBoom;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<UCameraComponent> FollowCamera;

	/** First-person view through helmet camera 0 (V key). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<UCameraComponent> HelmetViewCamera;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<USmartHelmetComponent> Helmet;

	/** Skeleton bone the helmet is attached to. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet")
	FName HeadBone = TEXT("head");

	/** Helmet centre relative to the head bone, in character axes (X fwd, Y right, Z up), cm. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet")
	FVector HelmetOffset = FVector(1.5f, 0.f, 14.f);

	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UInputMappingContext> DefaultMappingContext;

	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UInputMappingContext> MouseLookMappingContext;

	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UInputAction> JumpAction;

	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UInputAction> MoveAction;

	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UInputAction> LookAction;

	UPROPERTY(EditAnywhere, Category = "Input")
	TObjectPtr<UInputAction> MouseLookAction;

	/** Test motion: operator walks a circle (head moves during unsynchronised triggers). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Motion")
	bool bAutoWalk = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Motion", meta = (ClampMin = 0, ClampMax = 1))
	float AutoWalkInput = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "SmartHelmet|Motion")
	float AutoTurnDegPerSec = 25.f;

	UFUNCTION(BlueprintCallable, Category = "SmartHelmet")
	void AlignHelmetToHead();

	virtual void Tick(float DeltaSeconds) override;

	UFUNCTION(BlueprintCallable, Category = "SmartHelmet")
	void ToggleHelmetView();

	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void PostInitializeComponents() override;

protected:
	virtual void NotifyControllerChanged() override;
	virtual void SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) override;

	void Move(const FInputActionValue& Value);
	void Look(const FInputActionValue& Value);
	void OnCapture();
	void OnToggleAuto();

	UPROPERTY(Transient)
	TObjectPtr<UInputMappingContext> HelmetMappingContext;

	UPROPERTY(Transient)
	TObjectPtr<UInputAction> CaptureAction;

	UPROPERTY(Transient)
	TObjectPtr<UInputAction> ToggleAutoAction;

	UPROPERTY(Transient)
	TObjectPtr<UInputAction> ViewAction;
};
