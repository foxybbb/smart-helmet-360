#include "SmartHelmet/SmartHelmetCharacter.h"

#include "AnimationRuntime.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Engine/Engine.h"
#include "Engine/LocalPlayer.h"
#include "Engine/SkeletalMesh.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpringArmComponent.h"
#include "InputAction.h"
#include "InputActionValue.h"
#include "InputMappingContext.h"
#include "SmartHelmet/SmartHelmetComponent.h"
#include "UObject/ConstructorHelpers.h"

ASmartHelmetCharacter::ASmartHelmetCharacter()
{
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.f);
	bUseControllerRotationPitch = bUseControllerRotationYaw = bUseControllerRotationRoll = false;

	UCharacterMovementComponent* Move = GetCharacterMovement();
	Move->bOrientRotationToMovement = true;
	Move->RotationRate = FRotator(0.f, 500.f, 0.f);
	Move->JumpZVelocity = 500.f;
	Move->AirControl = 0.35f;
	Move->MaxWalkSpeed = 300.f;          // walking operator
	Move->MinAnalogWalkSpeed = 20.f;
	Move->BrakingDecelerationWalking = 2000.f;

	// Mannequin from the UE template content (/Game/Characters)
	static ConstructorHelpers::FObjectFinder<USkeletalMesh> MeshF(TEXT("/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple.SKM_Manny_Simple"));
	static ConstructorHelpers::FClassFinder<UAnimInstance> AnimF(TEXT("/Game/Characters/Mannequins/Anims/Unarmed/ABP_Unarmed"));
	if (MeshF.Succeeded()) { GetMesh()->SetSkeletalMeshAsset(MeshF.Object); }
	if (AnimF.Succeeded()) { GetMesh()->SetAnimInstanceClass(AnimF.Class); }
	GetMesh()->SetRelativeLocationAndRotation(FVector(0, 0, -97.f), FRotator(0, -90.f, 0));

	static ConstructorHelpers::FObjectFinder<UInputMappingContext> ImcF(TEXT("/Game/Input/IMC_Default.IMC_Default"));
	static ConstructorHelpers::FObjectFinder<UInputMappingContext> ImcMouseF(TEXT("/Game/Input/IMC_MouseLook.IMC_MouseLook"));
	static ConstructorHelpers::FObjectFinder<UInputAction> JumpF(TEXT("/Game/Input/Actions/IA_Jump.IA_Jump"));
	static ConstructorHelpers::FObjectFinder<UInputAction> MoveF(TEXT("/Game/Input/Actions/IA_Move.IA_Move"));
	static ConstructorHelpers::FObjectFinder<UInputAction> LookF(TEXT("/Game/Input/Actions/IA_Look.IA_Look"));
	static ConstructorHelpers::FObjectFinder<UInputAction> MouseLookF(TEXT("/Game/Input/Actions/IA_MouseLook.IA_MouseLook"));
	DefaultMappingContext = ImcF.Object;
	MouseLookMappingContext = ImcMouseF.Object;
	JumpAction = JumpF.Object;
	MoveAction = MoveF.Object;
	LookAction = LookF.Object;
	MouseLookAction = MouseLookF.Object;

	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->TargetArmLength = 320.f;
	CameraBoom->SocketOffset = FVector(0, 40.f, 40.f);
	CameraBoom->bUsePawnControlRotation = true;

	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName);
	FollowCamera->bUsePawnControlRotation = false;

	Helmet = CreateDefaultSubobject<USmartHelmetComponent>(TEXT("SmartHelmet"));
	Helmet->SetupAttachment(GetMesh(), HeadBone);

	HelmetViewCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("HelmetViewCamera"));
	HelmetViewCamera->SetupAttachment(Helmet);
	HelmetViewCamera->SetRelativeLocation(FVector(12.f, 0, 0));
	HelmetViewCamera->FieldOfView = 100.f;
	HelmetViewCamera->SetAutoActivate(false);
}

void ASmartHelmetCharacter::AlignHelmetToHead()
{
	USkeletalMesh* SM = GetMesh() ? GetMesh()->GetSkeletalMeshAsset() : nullptr;
	if (!SM || !Helmet) { return; }
	const FReferenceSkeleton& Ref = SM->GetRefSkeleton();
	const int32 Bone = Ref.FindBoneIndex(HeadBone);
	if (Bone == INDEX_NONE)
	{
		Helmet->AttachToComponent(GetMesh(), FAttachmentTransformRules::KeepRelativeTransform);
		Helmet->SetRelativeLocationAndRotation(FVector(0, 0, 175.f), FRotator(0, 90.f, 0));
		return;
	}
	// Mesh component space: the mannequin faces +Y, so the helmet (X fwd) needs yaw +90 there.
	const FTransform HeadCS = FAnimationRuntime::GetComponentSpaceTransformRefPose(Ref, Bone);
	const FVector OffsetCS(HelmetOffset.Y * -1.f, HelmetOffset.X, HelmetOffset.Z); // char (fwd,right,up) -> mesh (+Y fwd, -X right)
	const FTransform DesiredCS(FRotator(0, 90.f, 0), HeadCS.GetLocation() + OffsetCS);
	Helmet->AttachToComponent(GetMesh(), FAttachmentTransformRules::KeepRelativeTransform, HeadBone);
	Helmet->SetRelativeTransform(DesiredCS.GetRelativeTransform(HeadCS));
}

void ASmartHelmetCharacter::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (bAutoWalk)
	{
		const float Yaw = GetActorRotation().Yaw + AutoTurnDegPerSec * DeltaSeconds;
		AddMovementInput(FRotator(0, Yaw, 0).Vector(), AutoWalkInput);
	}
}

void ASmartHelmetCharacter::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	AlignHelmetToHead();
}

void ASmartHelmetCharacter::PostInitializeComponents()
{
	Super::PostInitializeComponents();
	AlignHelmetToHead();

	// Helmet-specific actions are created at runtime, no extra assets needed.
	CaptureAction = NewObject<UInputAction>(this, TEXT("IA_HelmetCapture"));
	CaptureAction->ValueType = EInputActionValueType::Boolean;
	ToggleAutoAction = NewObject<UInputAction>(this, TEXT("IA_HelmetAuto"));
	ToggleAutoAction->ValueType = EInputActionValueType::Boolean;
	ViewAction = NewObject<UInputAction>(this, TEXT("IA_HelmetView"));
	ViewAction->ValueType = EInputActionValueType::Boolean;
	HelmetMappingContext = NewObject<UInputMappingContext>(this, TEXT("IMC_Helmet"));
	HelmetMappingContext->MapKey(CaptureAction, EKeys::C);
	HelmetMappingContext->MapKey(ToggleAutoAction, EKeys::T);
	HelmetMappingContext->MapKey(ViewAction, EKeys::V);
}

void ASmartHelmetCharacter::NotifyControllerChanged()
{
	Super::NotifyControllerChanged();
	if (APlayerController* PC = Cast<APlayerController>(Controller))
	{
		if (UEnhancedInputLocalPlayerSubsystem* Sub = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PC->GetLocalPlayer()))
		{
			if (DefaultMappingContext) { Sub->AddMappingContext(DefaultMappingContext, 0); }
			if (MouseLookMappingContext) { Sub->AddMappingContext(MouseLookMappingContext, 0); }
			if (HelmetMappingContext) { Sub->AddMappingContext(HelmetMappingContext, 1); }
		}
	}
}

void ASmartHelmetCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	UEnhancedInputComponent* EIC = Cast<UEnhancedInputComponent>(PlayerInputComponent);
	if (!EIC) { return; }
	if (JumpAction)
	{
		EIC->BindAction(JumpAction, ETriggerEvent::Started, this, &ACharacter::Jump);
		EIC->BindAction(JumpAction, ETriggerEvent::Completed, this, &ACharacter::StopJumping);
	}
	if (MoveAction) { EIC->BindAction(MoveAction, ETriggerEvent::Triggered, this, &ASmartHelmetCharacter::Move); }
	if (LookAction) { EIC->BindAction(LookAction, ETriggerEvent::Triggered, this, &ASmartHelmetCharacter::Look); }
	if (MouseLookAction) { EIC->BindAction(MouseLookAction, ETriggerEvent::Triggered, this, &ASmartHelmetCharacter::Look); }
	if (CaptureAction) { EIC->BindAction(CaptureAction, ETriggerEvent::Started, this, &ASmartHelmetCharacter::OnCapture); }
	if (ToggleAutoAction) { EIC->BindAction(ToggleAutoAction, ETriggerEvent::Started, this, &ASmartHelmetCharacter::OnToggleAuto); }
	if (ViewAction) { EIC->BindAction(ViewAction, ETriggerEvent::Started, this, &ASmartHelmetCharacter::ToggleHelmetView); }
}

void ASmartHelmetCharacter::Move(const FInputActionValue& Value)
{
	const FVector2D V = Value.Get<FVector2D>();
	if (!Controller) { return; }
	const FRotator Yaw(0, Controller->GetControlRotation().Yaw, 0);
	AddMovementInput(FRotationMatrix(Yaw).GetUnitAxis(EAxis::X), V.Y);
	AddMovementInput(FRotationMatrix(Yaw).GetUnitAxis(EAxis::Y), V.X);
}

void ASmartHelmetCharacter::Look(const FInputActionValue& Value)
{
	const FVector2D V = Value.Get<FVector2D>();
	AddControllerYawInput(V.X);
	AddControllerPitchInput(V.Y);
}

void ASmartHelmetCharacter::OnCapture()
{
	if (Helmet) { Helmet->CaptureFrame(); }
}

void ASmartHelmetCharacter::OnToggleAuto()
{
	if (!Helmet) { return; }
	Helmet->SetAutoCapture(!Helmet->bAutoCapture, Helmet->CaptureInterval);
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(9002, 3.f, FColor::Yellow, Helmet->bAutoCapture
			? FString::Printf(TEXT("SmartHelmet: auto capture ON, every %.1f s"), Helmet->CaptureInterval)
			: FString(TEXT("SmartHelmet: auto capture OFF")));
	}
}

void ASmartHelmetCharacter::ToggleHelmetView()
{
	const bool bHelmet = !HelmetViewCamera->IsActive();
	HelmetViewCamera->SetActive(bHelmet);
	FollowCamera->SetActive(!bHelmet);
}
