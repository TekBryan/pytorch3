import logging as log

from sqlalchemy.orm import Session

from db.models.AIModels import AIModels, UsersAIModels


def addAIModel(path: str, email: str, label: str, user_id: int, db: Session, acc: any, loss: any):
    log.info("Adding New AI Model")
    try:
        aimodel = AIModels(path = path, created_by = email, label = label, accuracy = acc, loss = loss)
        db.add(aimodel)

        db.commit()

        user_aimodel = UsersAIModels(user_id = user_id, aimodel_id = aimodel.id, created_by = email)
        db.add(user_aimodel)

        db.commit()
    except Exception as e:
        db.rollback()
        log.error(e)
        raise e
    
def updateAIModel(aimodel_id: int, db: Session, acc: any, loss: any):
    log.info("Updating aimodel values in db")
    try:
        aimodel = db.query(AIModels).filter(AIModels.id == aimodel_id).first()
        if aimodel is None:
            raise ValueError(f"No AI model found with id: {aimodel_id}")
        
        # Update the accuracy and loss
        aimodel.accuracy = acc
        aimodel.loss = loss

        # Commit the changes to the database
        db.commit()
    except Exception as e:
        db.rollback()
        log.error(e)
        raise e