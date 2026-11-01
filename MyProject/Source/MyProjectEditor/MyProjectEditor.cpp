#include "Modules/ModuleManager.h"
#include "SmartHelmetToolset.h"
#include "ToolsetRegistry/UToolsetRegistry.h"

/** Registers the Smart Helmet toolset with the UE MCP server (ToolsetRegistry). */
class FMyProjectEditorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		UToolsetRegistry::RegisterToolsetClass(USmartHelmetToolset::StaticClass());
	}

	virtual void ShutdownModule() override
	{
		if (UObjectInitialized())
		{
			UToolsetRegistry::UnregisterToolsetClass(USmartHelmetToolset::StaticClass());
		}
	}
};

IMPLEMENT_MODULE(FMyProjectEditorModule, MyProjectEditor);
