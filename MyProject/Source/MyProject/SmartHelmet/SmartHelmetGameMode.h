#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "SmartHelmetGameMode.generated.h"

/** Spawns the Smart Helmet operator as the player pawn. */
UCLASS()
class MYPROJECT_API ASmartHelmetGameMode : public AGameModeBase
{
	GENERATED_BODY()
public:
	ASmartHelmetGameMode();
};
